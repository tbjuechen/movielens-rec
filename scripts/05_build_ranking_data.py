"""Offline generation of ranking training samples using recall-based candidate sets.

Two-phase pipeline:
  Phase 1 (CPU, multiprocessing): ItemCF + Pop + Genre recall per user snapshot
  Phase 2 (GPU, batch): DualTower recall per snapshot → merge into Phase 1 candidates

This aligns ranking training distribution with online inference.
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
RECALL_K = 100
MERGE_K = 100
MAX_INTERACTIONS_PER_USER = 100
SAMPLES_PER_INTERACTION = 4
RECALL_WORKERS = 32
DT_BATCH = 4096


# ── Module-level shared data for multiprocessing (fork COW) ──
_shared = {}


def _build_user_snapshots(uid):
    """Worker: build feature snapshots and CPU recall candidates for one user.

    Returns:
        interactions_out: list of (uid, mid, rating, snap_idx, channels_dict, watched_list)
            where channels_dict has per-channel candidate lists (watched filtered)
        snapshots_out: list of (uid, snap_idx, history_list, top_genres_list)
            unique snapshots needed for DT recall
    """
    s = _shared
    interactions = s['user_interactions'][uid]  # list of (movieId, rating, genres_str, timestamp)
    item_cf = s['item_cf']
    icf_ready = s['icf_ready']
    genre_recall = s['genre_recall']
    pop_candidates = s['pop']
    merger = s['merger']
    weights = s['weights']

    interactions_out = []
    snapshots_out = []
    history = []
    history_ts = []  # timestamps for ts_diff computation
    genre_counter = Counter()

    n_total = len(interactions)
    sample_start = max(0, n_total - MAX_INTERACTIONS_PER_USER)
    cur_snap_idx = -1

    for i, (mid, rating, genres_str, timestamp) in enumerate(interactions):
        # Update snapshot every SNAPSHOT_WINDOW interactions
        if i % SNAPSHOT_WINDOW == 0:
            snap_history = list(history[-USER_HISTORY_MAX_LEN:])
            snap_top_genres = [g for g, _ in genre_counter.most_common(USER_TOP_GENRES_MAX_LEN)]
            # Compute ts_diff: (latest_ts - each_ts) / 3600
            if history_ts:
                recent_ts = history_ts[-USER_HISTORY_MAX_LEN:]
                max_ts = recent_ts[-1]
                snap_ts_diff = [(max_ts - t) / 3600.0 for t in recent_ts]
            else:
                snap_ts_diff = []
            cur_snap_idx = i // SNAPSHOT_WINDOW
            if i >= sample_start:
                snapshots_out.append((uid, cur_snap_idx, snap_history, snap_top_genres, snap_ts_diff))

        if i >= sample_start:
            watched_set = set(history)
            channels = {}

            if icf_ready and snap_history:
                channels['item_cf'] = item_cf.retrieve(snap_history, k=RECALL_K)
            channels['popularity'] = pop_candidates
            if snap_top_genres:
                channels['genre'] = genre_recall.retrieve(snap_top_genres, k=RECALL_K)

            # Filter watched per channel (keep separate for Phase 3 multi-channel merge)
            for name in list(channels.keys()):
                channels[name] = [iid for iid in channels[name] if iid not in watched_set][:RECALL_K]

            interactions_out.append((uid, mid, rating, cur_snap_idx, channels, list(watched_set)))

        # Update history (always)
        history.append(mid)
        history_ts.append(timestamp)
        if genres_str:
            for g in genres_str.split('|'):
                genre_counter[g] += 1

    return interactions_out, snapshots_out


def main():
    print("=== Building Ranking Training Data (Recall-Based Candidates) ===")

    # 1. Load data
    print("[1/7] Loading data...")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    print(f"  train_data: {len(train_data):,} interactions")

    raw_dir = Path(PROCESSED_DATA_DIR).parent / "raw" / "ml-32m"
    movies = pd.read_csv(raw_dir / "movies.csv")
    movie_genres = dict(zip(movies['movieId'].values, movies['genres'].values))

    # 2. Load CPU recall models (before CUDA init for fork safety)
    print("[2/7] Loading recall models...")
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

    # 3. Prepare per-user interaction lists
    print("[3/7] Preparing per-user interactions...")
    train_data = train_data.sort_values(['userId', 'timestamp'])
    train_data['genres_str'] = train_data['movieId'].map(movie_genres).fillna('')

    user_interactions = {}
    for uid, group in tqdm(train_data.groupby('userId'), desc="Grouping"):
        user_interactions[uid] = list(zip(
            group['movieId'].values,
            group['rating'].values,
            group['genres_str'].values,
            group['timestamp'].values,
        ))
    eval_users = list(user_interactions.keys())
    print(f"  {len(eval_users):,} users, {len(train_data):,} interactions")

    # ================================================================
    # Phase 1: Multiprocessing CPU recall (ItemCF + Pop + Genre)
    # Must fork BEFORE any CUDA init to avoid deadlock
    # ================================================================
    print(f"[4/7] Phase 1: Parallel CPU recall ({RECALL_WORKERS} workers)...")
    _shared['user_interactions'] = user_interactions
    _shared['item_cf'] = item_cf
    _shared['icf_ready'] = icf_ready
    _shared['genre_recall'] = genre_recall
    _shared['pop'] = pop_candidates
    _shared['merger'] = merger
    _shared['weights'] = MERGER_WEIGHTS

    all_interactions = []  # (uid, mid, rating, snap_idx, cpu_merged, watched_list)
    all_snapshots = []     # (uid, snap_idx, history_list, top_genres_list)
    with Pool(RECALL_WORKERS) as pool:
        for interactions_out, snapshots_out in tqdm(
                pool.imap(_build_user_snapshots, eval_users, chunksize=64),
                total=len(eval_users), desc="CPU Recall"):
            all_interactions.extend(interactions_out)
            all_snapshots.extend(snapshots_out)
    _shared.clear()

    print(f"  Phase 1 done: {len(all_interactions):,} interactions, {len(all_snapshots):,} snapshots")

    # ================================================================
    # Phase 2: DualTower recall (GPU batch, after fork)
    # ================================================================
    print("[5/7] Phase 2: DualTower recall (GPU batch)...")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"  Device: {device}")

    # Load encoder + profiles for DT
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()

    item_profile['movieId_encoded'] = encoder.transform_categorical(item_profile['movieId'], 'movieId')
    item_profile['tmdb_genres_encoded'] = encoder.transform_categorical(item_profile['tmdb_genres'], 'genres', is_list=True, max_len=ITEM_GENRES_MAX_LEN)
    item_cont = encoder.transform_continuous(item_profile, ['release_year_orig', 'avg_rating', 'revenue', 'budget', 'vote_count_ml'], prefix="item")
    item_profile['release_year_norm'] = item_cont['item_release_year_orig']
    item_profile['avg_rating_norm'] = item_cont['item_avg_rating']
    item_profile['revenue_norm'] = item_cont['item_revenue']

    dt_model = DualTowerModel(
        vocab_sizes=encoder.vocab_sizes, embed_dim=EMBEDDING_DIM, tau=TAU,
        time_decay_lambda=TIME_DECAY_LAMBDA, bpr_gamma=BPR_GAMMA, bpr_margin=BPR_MARGIN,
        loss_infonce_weight=LOSS_INFONCE_WEIGHT, loss_bpr_weight=LOSS_BPR_WEIGHT,
        logit_scale_max=LOGIT_SCALE_MAX, cont_bucket_size=CONT_BUCKET_SIZE
    ).to(device)

    dt_ready = False
    dt_path = Path(MODEL_WEIGHTS_DIR) / "dual_tower.pth"
    if dt_path.exists():
        dt_model.load_state_dict(torch.load(dt_path, map_location=device))
        dt_ready = True
        print("  Loaded DualTower model")
    dt_model.eval()

    dt_snap_results = {}  # (uid, snap_idx) → [movieId, ...]
    if dt_ready:
        # Build FAISS index
        print("  Building FAISS index...")
        item_profile_sorted = item_profile.sort_values('movieId_encoded')
        item_feat_tensors = {
            'item_id': torch.tensor(item_profile_sorted['movieId_encoded'].values, dtype=torch.long).to(device),
            'release_year': torch.tensor(item_profile_sorted['release_year_norm'].values, dtype=torch.float32).to(device),
            'avg_rating': torch.tensor(item_profile_sorted['avg_rating_norm'].values, dtype=torch.float32).to(device),
            'revenue': torch.tensor(item_profile_sorted['revenue_norm'].values, dtype=torch.float32).to(device),
            'tmdb_genres': torch.tensor(np.stack(item_profile_sorted['tmdb_genres_encoded'].values), dtype=torch.long).to(device),
        }
        with torch.no_grad():
            _, all_item_embs = dt_model(None, item_feat_tensors)
            all_item_embs = all_item_embs.cpu().numpy().astype('float32')
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(all_item_embs)
        idx_to_movie_id = item_profile_sorted['movieId'].values
        print(f"  FAISS index: {index.ntotal} items")

        # Prepare snapshot features for batch DT forward
        movie_vocab = encoder.vocabularies.get('movieId', {'<PAD>': 0})
        genre_vocab = encoder.vocabularies.get('genres', {'<PAD>': 0})

        user_profile['userId_encoded'] = encoder.transform_categorical(user_profile['userId'], 'userId')
        user_cont = encoder.transform_continuous(user_profile, ['avg_rating', 'activity'], prefix="user")
        user_profile['avg_rating_norm'] = user_cont['user_avg_rating']
        user_profile['activity_norm'] = user_cont['user_activity']

        max_uid = int(user_profile['userId'].max())
        u_idx = user_profile['userId'].values
        user_encoded_id = np.zeros(max_uid + 1, dtype=np.int64)
        user_avg_rating_arr = np.zeros(max_uid + 1, dtype=np.float32)
        user_activity_arr = np.zeros(max_uid + 1, dtype=np.float32)
        user_encoded_id[u_idx] = user_profile['userId_encoded'].values
        user_avg_rating_arr[u_idx] = user_profile['avg_rating_norm'].values
        user_activity_arr[u_idx] = user_profile['activity_norm'].values

        # Encode all snapshot history/genres/ts_diff
        snap_keys = []
        snap_history_enc = np.zeros((len(all_snapshots), USER_HISTORY_MAX_LEN), dtype=np.int64)
        snap_top_genres_enc = np.zeros((len(all_snapshots), USER_TOP_GENRES_MAX_LEN), dtype=np.int64)
        snap_ts_diff_arr = np.zeros((len(all_snapshots), USER_HISTORY_MAX_LEN), dtype=np.float32)
        snap_uids = np.zeros(len(all_snapshots), dtype=np.int64)

        for si, (uid, snap_idx, hist, top_genres, ts_diff) in enumerate(all_snapshots):
            snap_keys.append((uid, snap_idx))
            snap_uids[si] = uid
            for j, mid in enumerate(hist[-USER_HISTORY_MAX_LEN:]):
                snap_history_enc[si, j] = movie_vocab.get(mid, 0)
            for j, g in enumerate(top_genres[:USER_TOP_GENRES_MAX_LEN]):
                snap_top_genres_enc[si, j] = genre_vocab.get(g, 0)
            for j, td in enumerate(ts_diff[-USER_HISTORY_MAX_LEN:]):
                snap_ts_diff_arr[si, j] = td

        # Batch forward + FAISS search
        n_snaps = len(snap_keys)
        print(f"  Batch DT forward ({n_snaps:,} snapshots, batch={DT_BATCH})...")
        for start in tqdm(range(0, n_snaps, DT_BATCH), desc="DT Recall"):
            end = min(start + DT_BATCH, n_snaps)
            batch_uids = snap_uids[start:end]
            user_tensor = {
                'user_id': torch.from_numpy(user_encoded_id[batch_uids]).to(device),
                'avg_rating': torch.from_numpy(user_avg_rating_arr[batch_uids]).to(device),
                'activity': torch.from_numpy(user_activity_arr[batch_uids]).to(device),
                'history': torch.from_numpy(snap_history_enc[start:end]).to(device),
                'history_ts_diff': torch.from_numpy(snap_ts_diff_arr[start:end]).to(device),
                'top_genres': torch.from_numpy(snap_top_genres_enc[start:end]).to(device),
            }
            with torch.no_grad():
                u_embs, _ = dt_model(user_tensor, None)
                u_embs = u_embs.cpu().numpy().astype('float32')
            _, I = index.search(u_embs, RECALL_K)
            for i in range(end - start):
                dt_snap_results[snap_keys[start + i]] = [int(idx_to_movie_id[j]) for j in I[i]]

        print(f"  DT recall done: {len(dt_snap_results):,} snapshot results")
    else:
        print("  WARNING: DualTower not found, skipping DT recall")

    # ================================================================
    # Phase 3: Merge CPU + DT candidates → Create Candidate Pool
    # ================================================================
    print("[6/7] Merging candidates into pools...")
    pool_data = []
    final_merger = RecallMerger(top_k=MERGE_K)

    for uid, mid, rating, snap_idx, channels, watched_list in tqdm(all_interactions, desc="Create Pool"):
        # Add DT as another channel
        if dt_ready:
            dt_cands = dt_snap_results.get((uid, snap_idx), [])
            if dt_cands:
                watched_set = set(watched_list)
                channels['dual_tower'] = [iid for iid in dt_cands if iid not in watched_set][:RECALL_K]

        # RRF merge
        merged = final_merger.merge(channels, weights=MERGER_WEIGHTS) if channels else []
        if not merged:
            continue

        # Keep Top 100 as the negative candidate pool (excluding the positive item)
        pool = [int(c) for c in merged if c != mid][:100]
        if len(pool) < 10: # Skip if too few candidates
            continue

        ctr_label = 1.0 if rating >= 3.0 else 0.0
        rating_norm = rating / 5.0
        pool_data.append((uid, mid, ctr_label, rating_norm, pool))

    print(f"  Total positive interactions with pools: {len(pool_data):,}")

    # 7. Save as Pool format
    print("[7/7] Saving Candidate Pool...")
    df = pd.DataFrame(pool_data, columns=['userId', 'movieId', 'ctr_label', 'rating_norm', 'candidate_pool'])
    
    out_path = Path(PROCESSED_DATA_DIR) / "ranking_candidate_pool.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n=== Done ===\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
