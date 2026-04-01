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
    MERGER_WEIGHTS, RANK_TRAIN_POOL_SIZE, RANK_EVAL_POOL_SIZE,
    RANK_FORCE_INSERT_TARGET, RANK_EXPLICIT_NEGATIVE_THRESHOLD,
)
from src.features.encoder import FeatureEncoder
from src.models.recall.dual_tower import DualTowerModel
from src.models.recall.item_cf import ItemCFModel
from src.models.recall.simple_recall import PopularityRecall, GenreRecall
from src.models.recall.merger import RecallMerger

SNAPSHOT_WINDOW = 10
RECALL_K = max(RANK_EVAL_POOL_SIZE, 500)
MERGE_K = max(RANK_EVAL_POOL_SIZE, 500)
MAX_INTERACTIONS_PER_USER = 100
SAMPLES_PER_INTERACTION = 4
RECALL_WORKERS = 32
DT_BATCH = 4096
MIN_USER_TOTAL_INTERACTIONS = 6


# ── Module-level shared data for multiprocessing (fork COW) ──
_shared = {}


def _filter_valid_users(train_data, val_data, test_data):
    total_counts = pd.concat([
        train_data[['userId']],
        val_data[['userId']],
        test_data[['userId']],
    ], ignore_index=True)['userId'].value_counts()
    valid_user_ids = set(total_counts[total_counts >= MIN_USER_TOTAL_INTERACTIONS].index.tolist())
    return valid_user_ids


def _unique_int_list(values):
    seen = set()
    out = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _prepare_candidate_pool(merged_candidates, actual_list, pool_size, force_insert=True):
    actual_items = [int(x) for x in actual_list] if actual_list else []
    if not actual_items:
        return None, {
            'target_in_pool_before_fix': False,
            'target_force_inserted': False,
            'target_pool_was_empty': False,
        }

    deduped = []
    seen = set()
    for item in merged_candidates:
        item = int(item)
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
        if len(deduped) >= pool_size:
            break

    actual_set = set(actual_items)
    target_in_pool = bool(actual_set & set(deduped))
    pool_was_empty = len(deduped) == 0
    force_inserted = False

    if force_insert and not target_in_pool:
        target_item = actual_items[0]
        if deduped:
            deduped[-1] = target_item
        else:
            deduped = [target_item]
            pool_was_empty = True
        force_inserted = True

    return deduped, {
        'target_in_pool_before_fix': target_in_pool,
        'target_force_inserted': force_inserted,
        'target_pool_was_empty': pool_was_empty,
    }


def _merge_with_dt(pool_merger, cpu_merged, dt_snap_results, uid, snap_idx, watched_items):
    dt_cands = dt_snap_results.get((uid, snap_idx), [])
    if dt_cands:
        watched_set = set(watched_items)
        dt_cands = [c for c in dt_cands if c not in watched_set][:RECALL_K]
        return pool_merger.merge(
            {'cpu': cpu_merged, 'dt': dt_cands},
            weights={'cpu': 1.0, 'dt': MERGER_WEIGHTS.get('dual_tower', 1.0)}
        )
    return cpu_merged


def _build_user_data(uid):
    """Worker: Build EVERYTHING for one user (Recall + Merge + Training/Eval data)."""
    s = _shared
    interactions = s['user_interactions'][uid] 
    item_cf = s['item_cf']
    icf_ready = s['icf_ready']
    genre_recall = s['genre_recall']
    pop_candidates = s['pop']
    merger = s['merger']
    weights = s['weights']
    val_gt = s['val_gt']
    test_gt = s['test_gt']

    train_out = []
    val_out = []
    test_out = []
    snapshots_out = [] # Needed for Phase 2 GPU DT recall
    
    history = []
    history_ts = []
    genre_counter = Counter()

    n_total = len(interactions)
    sample_start = max(0, n_total - MAX_INTERACTIONS_PER_USER)

    for i, (mid, rating, genres_str, timestamp) in enumerate(interactions):
        # 1. Update Snapshot (Features)
        if i % SNAPSHOT_WINDOW == 0:
            snap_history = list(history[-USER_HISTORY_MAX_LEN:])
            snap_top_genres = [g for g, _ in genre_counter.most_common(USER_TOP_GENRES_MAX_LEN)]
            if history_ts:
                recent_ts = history_ts[-USER_HISTORY_MAX_LEN:]
                max_ts = recent_ts[-1]
                snap_ts_diff = [(max_ts - t) / 3600.0 for t in recent_ts]
            else:
                snap_ts_diff = []
            
            cur_snap_idx = i // SNAPSHOT_WINDOW
            if i >= sample_start:
                snapshots_out.append((uid, cur_snap_idx, snap_history, snap_top_genres, snap_ts_diff))

        # 2. Recall + Merge (If in sampling window)
        if i >= sample_start:
            watched_set = set(history)
            channels = {}
            if icf_ready and snap_history:
                channels['item_cf'] = [iid for iid in item_cf.retrieve(snap_history, k=RECALL_K) if iid not in watched_set][:RECALL_K]
            channels['popularity'] = [iid for iid in pop_candidates if iid not in watched_set][:RECALL_K]
            if snap_top_genres:
                channels['genre'] = [iid for iid in genre_recall.retrieve(snap_top_genres, k=RECALL_K) if iid not in watched_set][:RECALL_K]

            # RRF Merge (Initial CPU-only merge, DT will be added later if available)
            merged = merger.merge(channels, weights=weights) if channels else []
            
            # Record state for training, even if the current merged pool is empty.
            # The final integration step may still force-insert the target item.
            if rating >= RANK_EXPLICIT_NEGATIVE_THRESHOLD:
                ctr_label = 1.0
                rating_norm = rating / 5.0
                train_out.append({
                    'uid': uid, 'mid': mid, 'ctr': ctr_label, 'rat': rating_norm,
                    'cpu_merged': merged, 'watched': list(watched_set), 'snap_idx': i // SNAPSHOT_WINDOW
                })

            # Record state for Evaluation (If it's the last interaction)
            if i == n_total - 1:
                if uid in val_gt:
                    val_out.append({'userId': uid, 'cpu_merged': merged, 'actual': val_gt[uid], 'snap_idx': i // SNAPSHOT_WINDOW, 'watched': list(watched_set)})
                if uid in test_gt:
                    test_out.append({'userId': uid, 'cpu_merged': merged, 'actual': test_gt[uid], 'snap_idx': i // SNAPSHOT_WINDOW, 'watched': list(watched_set)})

        # Update History
        history.append(mid)
        history_ts.append(timestamp)
        if genres_str:
            for g in genres_str.split('|'):
                genre_counter[g] += 1

    return train_out, val_out, test_out, snapshots_out

def main():
    print("=== Building Ranking Data (Optimized Multiprocessing) ===")

    # 1. Load data
    print("[1/7] Loading data...")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    val_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "val_data.parquet")
    test_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "test_data.parquet")
    valid_user_ids = _filter_valid_users(train_data, val_data, test_data)
    print(f"  Ranking users with total interactions > 5: {len(valid_user_ids):,}")
    train_data = train_data[train_data['userId'].isin(valid_user_ids)].copy()
    val_data = val_data[val_data['userId'].isin(valid_user_ids)].copy()
    test_data = test_data[test_data['userId'].isin(valid_user_ids)].copy()
    explicit_negative_map = (
        train_data[train_data['rating'] < RANK_EXPLICIT_NEGATIVE_THRESHOLD]
        .groupby('userId')['movieId']
        .apply(lambda x: _unique_int_list(x.tolist()))
        .to_dict()
    )
    
    raw_dir = Path(PROCESSED_DATA_DIR).parent / "raw" / "ml-32m"
    movies = pd.read_csv(raw_dir / "movies.csv")
    movie_genres = dict(zip(movies['movieId'].values, movies['genres'].values))

    # 2. Load CPU models
    print("[2/7] Loading CPU recall models...")
    item_cf = ItemCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "item_sim_matrix.pkl")
    item_cf.load()
    pop_recall = PopularityRecall(item_profile_path=Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    genre_recall = GenreRecall(
        genre_to_items_path=Path(FEATURE_STORE_DIR) / "genre_to_items.json",
        item_profile_path=Path(PROCESSED_DATA_DIR) / "item_profile.parquet"
    )
    pop_candidates = pop_recall.retrieve(k=RECALL_K)

    # 3. Sort & Group
    print("[3/7] Preparing per-user interactions...")
    train_data = train_data.sort_values(['userId', 'timestamp'])
    train_data['genres_str'] = train_data['movieId'].map(movie_genres).fillna('')
    user_interactions = {uid: list(zip(g['movieId'].values, g['rating'].values, g['genres_str'].values, g['timestamp'].values))
                         for uid, g in tqdm(train_data.groupby('userId'), desc="Grouping")}
    
    val_gt = val_data.groupby('userId')['movieId'].apply(list).to_dict()
    test_gt = test_data.groupby('userId')['movieId'].apply(list).to_dict()

    # 4. Phase 1: Parallel CPU Recall + Merge
    print(f"[4/7] Phase 1: Parallel Recall & Merge ({RECALL_WORKERS} workers)...")
    _shared.update({
        'user_interactions': user_interactions, 'item_cf': item_cf, 'icf_ready': True,
        'genre_recall': genre_recall, 'pop': pop_candidates,
        'merger': RecallMerger(top_k=MERGE_K), 'weights': MERGER_WEIGHTS,
        'val_gt': val_gt, 'test_gt': test_gt
    })

    all_train = []; all_val = []; all_test = []; all_snapshots = []
    with Pool(RECALL_WORKERS) as pool:
        for t, v, te, s in tqdm(pool.imap(_build_user_data, list(user_interactions.keys()), chunksize=64),
                               total=len(user_interactions), desc="Parallel Build"):
            all_train.extend(t); all_val.extend(v); all_test.extend(te); all_snapshots.extend(s)
    _shared.clear()

    # 5. Phase 2: DualTower (GPU)
    print("[5/7] Phase 2: DualTower recall (GPU batch)...")
    # ... [Keep FAISS and DT logic, it's already fast] ...
    # Assume dt_snap_results is populated from batch DT search
    
    # [Rest of Phase 2 logic remains similar, getting dt_snap_results]


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

    snap_keys = []
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
    dt_snap_results = {}
    if dt_ready and n_snaps > 0:
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

    # ================================================================
    # Phase 3: Final Integration (Merge DT with CPU-only pools)
    # ================================================================
    print("[6/7] Integrating DualTower into candidate pools...")
    pool_merger = RecallMerger(top_k=MERGE_K)
    final_train = []
    final_val = []
    final_test = []
    pool_stats = {
        'train': {'target_in_pool_before_fix': 0, 'target_force_inserted': 0, 'target_pool_was_empty': 0},
        'val': {'target_in_pool_before_fix': 0, 'target_force_inserted': 0, 'target_pool_was_empty': 0},
        'test': {'target_in_pool_before_fix': 0, 'target_force_inserted': 0, 'target_pool_was_empty': 0},
    }

    # Process Train interactions
    for item in tqdm(all_train, desc="Final Train Pool"):
        uid, mid, snap_idx = item['uid'], item['mid'], item['snap_idx']
        merged = _merge_with_dt(pool_merger, item['cpu_merged'], dt_snap_results, uid, snap_idx, item['watched'])
        pool, stats = _prepare_candidate_pool(
            merged_candidates=merged,
            actual_list=[mid],
            pool_size=RANK_TRAIN_POOL_SIZE,
            force_insert=RANK_FORCE_INSERT_TARGET,
        )
        if pool is None:
            continue
        for key, value in stats.items():
            pool_stats['train'][key] += int(value)

        explicit_negatives = [iid for iid in explicit_negative_map.get(uid, []) if iid != int(mid)]
        neg_count = sum(1 for c in pool if c != int(mid))
        if neg_count >= 10:
            final_train.append((uid, mid, item['ctr'], item['rat'], pool, explicit_negatives))

    # Process Eval sets
    for item in tqdm(all_val, desc="Final Val Pool"):
        uid, snap_idx = item['userId'], item['snap_idx']
        actual_list = item['actual']
        merged = _merge_with_dt(pool_merger, item['cpu_merged'], dt_snap_results, uid, snap_idx, item['watched'])
        candidates, stats = _prepare_candidate_pool(
            merged_candidates=merged,
            actual_list=actual_list,
            pool_size=RANK_EVAL_POOL_SIZE,
            force_insert=RANK_FORCE_INSERT_TARGET,
        )
        if candidates is None:
            continue
        for key, value in stats.items():
            pool_stats['val'][key] += int(value)
        final_val.append({
            'userId': uid,
            'candidates': candidates,
            'actual': actual_list,
            **stats,
        })

    for item in tqdm(all_test, desc="Final Test Pool"):
        uid, snap_idx = item['userId'], item['snap_idx']
        actual_list = item['actual']
        merged = _merge_with_dt(pool_merger, item['cpu_merged'], dt_snap_results, uid, snap_idx, item['watched'])
        candidates, stats = _prepare_candidate_pool(
            merged_candidates=merged,
            actual_list=actual_list,
            pool_size=RANK_EVAL_POOL_SIZE,
            force_insert=RANK_FORCE_INSERT_TARGET,
        )
        if candidates is None:
            continue
        for key, value in stats.items():
            pool_stats['test'][key] += int(value)
        final_test.append({
            'userId': uid,
            'candidates': candidates,
            'actual': actual_list,
            **stats,
        })

    # 7. Save
    print("[7/7] Saving Results...")
    pd.DataFrame(final_train, columns=['userId', 'movieId', 'ctr_label', 'rating_norm', 'candidate_pool', 'explicit_negatives']).to_parquet(
        Path(PROCESSED_DATA_DIR) / "ranking_candidate_pool.parquet", index=False)
    pd.DataFrame(final_val).to_parquet(Path(PROCESSED_DATA_DIR) / "ranking_val_candidates.parquet", index=False)
    pd.DataFrame(final_test).to_parquet(Path(PROCESSED_DATA_DIR) / "ranking_test_candidates.parquet", index=False)
    
    print(f"\n=== Done ===\nSaved Training: {len(final_train):,}, Val: {len(final_val):,}, Test: {len(final_test):,}")
    for split, stats in pool_stats.items():
        print(
            f"  {split}: raw_hit={stats['target_in_pool_before_fix']:,}, "
            f"force_inserted={stats['target_force_inserted']:,}, "
            f"empty_pool_fallback={stats['target_pool_was_empty']:,}"
        )


if __name__ == "__main__":
    main()
