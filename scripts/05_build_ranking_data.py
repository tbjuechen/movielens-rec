"""Offline generation of ranking training samples using recall-based candidate sets.

For each training interaction, runs recall (ItemCF + Pop + Genre + DualTower)
using feature snapshots (updated every 10 interactions per user) to generate
candidates. This aligns ranking training distribution with online inference.
"""
import sys
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import (
    PROCESSED_DATA_DIR, FEATURE_STORE_DIR, MODEL_WEIGHTS_DIR,
    EMBEDDING_DIM, TAU, TIME_DECAY_LAMBDA, BPR_GAMMA, BPR_MARGIN,
    LOSS_INFONCE_WEIGHT, LOSS_BPR_WEIGHT, LOGIT_SCALE_MAX, CONT_BUCKET_SIZE,
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN,
    MERGER_WEIGHTS,
)
from src.features.encoder import FeatureEncoder
from src.models.recall.dual_tower import DualTowerModel
from src.models.recall.item_cf import ItemCFModel
from src.models.recall.simple_recall import PopularityRecall, GenreRecall
from src.models.recall.merger import RecallMerger

SNAPSHOT_WINDOW = 10
RECALL_K = 200
MERGE_K = 300
MAX_INTERACTIONS_PER_USER = 100
SAMPLES_PER_INTERACTION = 100
RECALL_WORKERS = 32


# ── Module-level shared data for multiprocessing (fork COW) ──
_shared = {}


def _build_user_snapshots(uid):
    """Worker: build feature snapshots and recall candidates for one user.

    Returns list of (userId, movieId, ctr_label, rating_norm, has_rating)
    for all interactions of this user.
    """
    s = _shared
    interactions = s['user_interactions'][uid]  # list of (movieId, rating, genres_str)
    item_cf = s['item_cf']
    icf_ready = s['icf_ready']
    genre_recall = s['genre_recall']
    pop_candidates = s['pop']
    merger = s['merger']
    weights = s['weights']
    movie_genres = s['movie_genres']

    samples = []
    history = []
    genre_counter = Counter()
    rng = np.random.RandomState(uid % (2**31))

    # Only generate samples for the last MAX_INTERACTIONS_PER_USER interactions
    n_total = len(interactions)
    sample_start = max(0, n_total - MAX_INTERACTIONS_PER_USER)

    for i, (mid, rating, genres_str) in enumerate(interactions):
        # Update snapshot every SNAPSHOT_WINDOW interactions
        if i % SNAPSHOT_WINDOW == 0:
            snap_history = list(history[-USER_HISTORY_MAX_LEN:])
            snap_top_genres = [g for g, _ in genre_counter.most_common(USER_TOP_GENRES_MAX_LEN)]

        # Only generate samples for the last N interactions
        if i >= sample_start:
            # Run recall with current snapshot
            watched_set = set(history)
            channels = {}

            if icf_ready and snap_history:
                channels['item_cf'] = item_cf.retrieve(snap_history, k=RECALL_K)
            channels['popularity'] = pop_candidates
            if snap_top_genres:
                channels['genre'] = genre_recall.retrieve(snap_top_genres, k=RECALL_K)

            # Filter watched
            for name in list(channels.keys()):
                channels[name] = [iid for iid in channels[name] if iid not in watched_set][:RECALL_K]

            merged = merger.merge(channels, weights=weights) if channels else []

            if merged:
                # Ensure target item is in candidate set
                merged_set = set(merged)
                if mid not in merged_set:
                    merged[-1] = mid

                # Sample negatives: pick SAMPLES_PER_INTERACTION-1 from candidates
                neg_candidates = [c for c in merged if c != mid]
                n_neg = min(SAMPLES_PER_INTERACTION - 1, len(neg_candidates))
                if n_neg < len(neg_candidates):
                    neg_sampled = list(rng.choice(neg_candidates, size=n_neg, replace=False))
                else:
                    neg_sampled = neg_candidates

                ctr_label = 1.0 if rating >= 3.0 else 0.0
                rating_norm = rating / 5.0

                # Positive sample
                samples.append((uid, mid, ctr_label, rating_norm, 1.0))
                # Negative samples
                for cand in neg_sampled:
                    samples.append((uid, cand, 0.0, 0.0, 0.0))

        # Update history (always, for all interactions)
        history.append(mid)
        if genres_str:
            for g in genres_str.split('|'):
                genre_counter[g] += 1

    return samples


def main():
    print("=== Building Ranking Training Data (Recall-Based Candidates) ===")

    # 1. Load data
    print("[1/5] Loading data...")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    print(f"  train_data: {len(train_data):,} interactions")

    # Load movie genres for snapshot genre counting
    raw_dir = Path(PROCESSED_DATA_DIR).parent / "raw" / "ml-32m"
    movies = pd.read_csv(raw_dir / "movies.csv")
    movie_genres = dict(zip(movies['movieId'].values, movies['genres'].values))

    # 2. Load recall models (before CUDA init for fork safety)
    print("[2/5] Loading recall models...")
    item_cf = ItemCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "item_sim_matrix.pkl")
    icf_ready = False
    try:
        item_cf.load()
        icf_ready = True
    except FileNotFoundError:
        print("  WARNING: ItemCF not found, skipping")

    pop_recall = PopularityRecall(item_profile_path=Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    genre_recall = GenreRecall(
        genre_to_items_path=Path(FEATURE_STORE_DIR) / "genre_to_items.json",
        item_profile_path=Path(PROCESSED_DATA_DIR) / "item_profile.parquet"
    )
    merger = RecallMerger(top_k=MERGE_K)
    pop_candidates = pop_recall.retrieve(k=RECALL_K)

    # 3. Prepare per-user interaction lists (sorted by timestamp)
    print("[3/5] Preparing per-user interactions...")
    train_data = train_data.sort_values(['userId', 'timestamp'])
    train_data['genres_str'] = train_data['movieId'].map(movie_genres).fillna('')

    user_interactions = {}
    for uid, group in tqdm(train_data.groupby('userId'), desc="Grouping"):
        user_interactions[uid] = list(zip(
            group['movieId'].values,
            group['rating'].values,
            group['genres_str'].values
        ))
    eval_users = list(user_interactions.keys())
    print(f"  {len(eval_users):,} users, {len(train_data):,} interactions")

    # 4. Phase 1: Multiprocessing recall (ItemCF + Pop + Genre)
    print(f"[4/5] Parallel recall ({RECALL_WORKERS} workers)...")
    _shared['user_interactions'] = user_interactions
    _shared['item_cf'] = item_cf
    _shared['icf_ready'] = icf_ready
    _shared['genre_recall'] = genre_recall
    _shared['pop'] = pop_candidates
    _shared['merger'] = merger
    _shared['weights'] = MERGER_WEIGHTS
    _shared['movie_genres'] = movie_genres

    all_samples = []
    with Pool(RECALL_WORKERS) as pool:
        for user_samples in tqdm(
                pool.imap(_build_user_snapshots, eval_users, chunksize=64),
                total=len(eval_users), desc="Recall"):
            all_samples.extend(user_samples)
    _shared.clear()

    print(f"  Phase 1 done: {len(all_samples):,} samples (before DT)")

    # 5. Save (DualTower recall added separately if needed)
    print("[5/5] Saving...")
    samples_arr = np.array(all_samples, dtype=np.float64)
    df = pd.DataFrame(samples_arr, columns=['userId', 'movieId', 'ctr_label', 'rating_norm', 'has_rating'])
    df['userId'] = df['userId'].astype(np.int64)
    df['movieId'] = df['movieId'].astype(np.int64)
    df['ctr_label'] = df['ctr_label'].astype(np.float32)
    df['rating_norm'] = df['rating_norm'].astype(np.float32)
    df['has_rating'] = df['has_rating'].astype(np.float32)

    out_path = Path(PROCESSED_DATA_DIR) / "ranking_train_samples.parquet"
    df.to_parquet(out_path, index=False)

    n_pos = int((df['ctr_label'] > 0.5).sum())
    n_neg = len(df) - n_pos
    print(f"\n=== Done ===")
    print(f"Total samples: {len(df):,} (pos={n_pos:,}, neg={n_neg:,})")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
