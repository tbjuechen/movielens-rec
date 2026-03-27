"""End-to-end ranking evaluation: recall candidates → ranking reorder → metrics."""
import argparse
import sys
from pathlib import Path
from collections import defaultdict
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
    RANK_ID_EMBED_DIM, RANK_GENRE_EMBED_DIM, RANK_CONT_EMBED_DIM,
    RANK_CONT_BUCKET_SIZE,
    RANK_CROSS_LAYERS, RANK_DROPOUT,
    RANK_NUM_EXPERTS, RANK_EXPERT_DIM, RANK_TOWER_DIMS,
    RANK_CTR_ALPHA, RANK_RATING_BETA, RANK_EVAL_KS,
)
from src.features.encoder import FeatureEncoder
from src.models.recall.dual_tower import DualTowerModel
from src.models.recall.item_cf import ItemCFModel
from src.models.recall.simple_recall import PopularityRecall, GenreRecall
from src.models.recall.merger import RecallMerger
from src.models.ranking.ranker import RankingModel
from src.evaluation.metrics import hitrate_at_k, ndcg_at_k, mrr


def apply_encoding(user_profile, item_profile, encoder):
    user_profile['userId_encoded'] = encoder.transform_categorical(user_profile['userId'], 'userId')
    user_profile['top_genres_encoded'] = encoder.transform_categorical(user_profile['top_genres'], 'genres', is_list=True, max_len=USER_TOP_GENRES_MAX_LEN)
    user_profile['history_encoded'] = encoder.transform_categorical(user_profile['history'], 'movieId', is_list=True, max_len=USER_HISTORY_MAX_LEN)

    item_profile['movieId_encoded'] = encoder.transform_categorical(item_profile['movieId'], 'movieId')
    item_profile['tmdb_genres_encoded'] = encoder.transform_categorical(item_profile['tmdb_genres'], 'genres', is_list=True, max_len=ITEM_GENRES_MAX_LEN)

    user_cont = encoder.transform_continuous(user_profile, ['avg_rating', 'activity'], prefix="user")
    user_profile['avg_rating_norm'] = user_cont['user_avg_rating']
    user_profile['activity_norm'] = user_cont['user_activity']

    item_cont = encoder.transform_continuous(item_profile, ['release_year_orig', 'avg_rating', 'revenue', 'budget', 'vote_count_ml'], prefix="item")
    item_profile['release_year_norm'] = item_cont['item_release_year_orig']
    item_profile['avg_rating_norm'] = item_cont['item_avg_rating']
    item_profile['revenue_norm'] = item_cont['item_revenue']
    item_profile['budget_norm'] = item_cont['item_budget']
    item_profile['vote_count_ml_norm'] = item_cont['item_vote_count_ml']
    return user_profile, item_profile


# Module-level shared data for multiprocessing (fork-based COW)
_shared = {}


def _recall_one_user(uid):
    """Worker: ItemCF + genre + pop + merge + filter watched → candidate list."""
    s = _shared
    watched_set = s['watched'][uid]

    channels = {}
    dt = s['dt_results'].get(uid)
    if dt is not None:
        channels['dual_tower'] = dt
    history = s['history'].get(uid)
    if s['icf_ready'] and history is not None:
        channels['item_cf'] = s['item_cf'].retrieve(list(history), k=s['RAW_K'])
    channels['popularity'] = s['pop']
    genres = s['genres'].get(uid)
    if genres is not None:
        channels['genre'] = s['genre_recall'].retrieve(list(genres), k=s['RAW_K'])

    for name in list(channels.keys()):
        channels[name] = [iid for iid in channels[name] if iid not in watched_set][:s['CH_K']]

    merged = s['merger'].merge(channels, weights=s['weights'])
    actual = s['ground_truth'][uid]
    return uid, merged, actual


def evaluate(test_mode=False):
    print("=== Ranking End-to-End Evaluation ===")

    MERGE_K = 500
    RAW_CHANNEL_K = 300
    CHANNEL_K = 200
    RECALL_WORKERS = 32

    # 1. Load data & encoder
    val_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "val_data.parquet")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")

    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    # 2. Load recall models
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
    dt_model.eval()

    item_cf = ItemCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "item_sim_matrix.pkl")
    icf_ready = False
    try:
        item_cf.load()
        icf_ready = True
    except FileNotFoundError:
        pass

    pop_recall = PopularityRecall(item_profile_path=Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    genre_recall = GenreRecall(
        genre_to_items_path=Path(FEATURE_STORE_DIR) / "genre_to_items.json",
        item_profile_path=Path(PROCESSED_DATA_DIR) / "item_profile.parquet"
    )
    merger = RecallMerger(top_k=MERGE_K)

    # 3. Build FAISS index
    index = None
    idx_to_movie_id = []
    if dt_ready:
        print("Building FAISS index...")
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
        print(f"FAISS index: {index.ntotal} items")

    # 4. Load ranking model
    ranking_model = RankingModel(
        vocab_sizes=encoder.vocab_sizes,
        id_embed_dim=RANK_ID_EMBED_DIM,
        genre_embed_dim=RANK_GENRE_EMBED_DIM,
        cont_embed_dim=RANK_CONT_EMBED_DIM,
        cont_bucket_size=RANK_CONT_BUCKET_SIZE,
        cross_layers=RANK_CROSS_LAYERS,
        dropout=RANK_DROPOUT,
        num_experts=RANK_NUM_EXPERTS,
        expert_dim=RANK_EXPERT_DIM,
        tower_dims=RANK_TOWER_DIMS,
    ).to(device)
    ranking_path = Path(MODEL_WEIGHTS_DIR) / "ranking_model.pth"
    if not ranking_path.exists():
        print(f"ERROR: {ranking_path} not found. Train the ranking model first.")
        return
    ranking_model.load_state_dict(torch.load(ranking_path, map_location=device, weights_only=True))
    ranking_model.eval()
    print("Loaded ranking model.")

    # 5. Prepare item feature lookup for ranking (indexed by raw movieId)
    max_iid = int(item_profile['movieId'].max())
    item_encoded_id = np.zeros(max_iid + 1, dtype=np.int64)
    item_release_year = np.zeros(max_iid + 1, dtype=np.float32)
    item_avg_rating = np.zeros(max_iid + 1, dtype=np.float32)
    item_revenue = np.zeros(max_iid + 1, dtype=np.float32)
    item_budget = np.zeros(max_iid + 1, dtype=np.float32)
    item_vote_count = np.zeros(max_iid + 1, dtype=np.float32)
    item_genres_arr = np.zeros((max_iid + 1, ITEM_GENRES_MAX_LEN), dtype=np.int64)

    i_idx = item_profile['movieId'].values
    item_encoded_id[i_idx] = item_profile['movieId_encoded'].values
    item_release_year[i_idx] = item_profile['release_year_norm'].values
    item_avg_rating[i_idx] = item_profile['avg_rating_norm'].values
    item_revenue[i_idx] = item_profile['revenue_norm'].values
    item_budget[i_idx] = item_profile['budget_norm'].values
    item_vote_count[i_idx] = item_profile['vote_count_ml_norm'].values
    item_genres_arr[i_idx] = np.stack(item_profile['tmdb_genres_encoded'].values)

    # 6. Prepare eval users & user lookup arrays
    val_ground_truth = val_data.groupby('userId')['movieId'].apply(list).to_dict()
    eval_users = [uid for uid in val_ground_truth if uid in set(user_profile['userId'].values)]
    if test_mode:
        eval_users = eval_users[:1000]
    print(f"Evaluating {len(eval_users)} users...")

    # User feature lookup arrays (for DT batch + ranking)
    max_uid = int(user_profile['userId'].max())
    u_idx = user_profile['userId'].values
    user_encoded_id = np.zeros(max_uid + 1, dtype=np.int64)
    user_avg_rating_arr = np.zeros(max_uid + 1, dtype=np.float32)
    user_activity_arr = np.zeros(max_uid + 1, dtype=np.float32)
    user_history_arr = np.zeros((max_uid + 1, USER_HISTORY_MAX_LEN), dtype=np.int64)
    user_ts_diff_arr = np.zeros((max_uid + 1, USER_HISTORY_MAX_LEN), dtype=np.float32)
    user_top_genres_arr = np.zeros((max_uid + 1, USER_TOP_GENRES_MAX_LEN), dtype=np.int64)

    user_encoded_id[u_idx] = user_profile['userId_encoded'].values
    user_avg_rating_arr[u_idx] = user_profile['avg_rating_norm'].values
    user_activity_arr[u_idx] = user_profile['activity_norm'].values
    user_history_arr[u_idx] = np.stack(user_profile['history_encoded'].values)
    user_top_genres_arr[u_idx] = np.stack(user_profile['top_genres_encoded'].values)
    for _, row in user_profile.iterrows():
        uid = row['userId']
        ts = row.get('history_ts_diff')
        if isinstance(ts, (list, np.ndarray)):
            ts = list(ts)[:USER_HISTORY_MAX_LEN]
            user_ts_diff_arr[uid, :len(ts)] = ts

    # Dicts for recall workers
    user_history_dict = dict(zip(user_profile['userId'].values, user_profile['history'].values))
    user_top_genres_dict = dict(zip(user_profile['userId'].values, user_profile['top_genres'].values))
    user_watched_dict = {}
    for uid_val, hist in user_history_dict.items():
        user_watched_dict[uid_val] = set(hist) if isinstance(hist, (list, np.ndarray)) else set()

    pop_candidates = pop_recall.retrieve(k=RAW_CHANNEL_K)

    # ================================================================
    # Phase 1: Batch DT recall (GPU, fast)
    # ================================================================
    dt_results = {}
    if dt_ready:
        print("[Phase 1] Batch dual-tower recall...")
        BATCH = 4096
        eval_uids_arr = np.array(eval_users, dtype=np.int64)
        for start in tqdm(range(0, len(eval_users), BATCH), desc="DualTower"):
            batch_uids = eval_uids_arr[start:start + BATCH]
            user_tensor = {
                'user_id': torch.from_numpy(user_encoded_id[batch_uids]).to(device),
                'avg_rating': torch.from_numpy(user_avg_rating_arr[batch_uids]).to(device),
                'activity': torch.from_numpy(user_activity_arr[batch_uids]).to(device),
                'history': torch.from_numpy(user_history_arr[batch_uids]).to(device),
                'history_ts_diff': torch.from_numpy(user_ts_diff_arr[batch_uids]).to(device),
                'top_genres': torch.from_numpy(user_top_genres_arr[batch_uids]).to(device),
            }
            with torch.no_grad():
                u_embs, _ = dt_model(user_tensor, None)
                u_embs = u_embs.cpu().numpy().astype('float32')
            _, I = index.search(u_embs, RAW_CHANNEL_K)
            for i, uid in enumerate(batch_uids):
                dt_results[int(uid)] = [int(idx_to_movie_id[j]) for j in I[i]]

    # ================================================================
    # Phase 2+3: Parallel recall (CPU) + Ranking (GPU) pipeline
    # Workers produce recall results, main process consumes for ranking.
    # pool.imap is lazy — GPU ranking runs while workers are still recalling.
    # ================================================================
    print(f"[Phase 2+3] Parallel recall ({RECALL_WORKERS} workers) + GPU ranking...")
    _shared['dt_results'] = dt_results
    _shared['item_cf'] = item_cf
    _shared['icf_ready'] = icf_ready
    _shared['genre_recall'] = genre_recall
    _shared['merger'] = merger
    _shared['pop'] = pop_candidates
    _shared['history'] = user_history_dict
    _shared['genres'] = user_top_genres_dict
    _shared['watched'] = user_watched_dict
    _shared['ground_truth'] = val_ground_truth
    _shared['weights'] = MERGER_WEIGHTS
    _shared['RAW_K'] = RAW_CHANNEL_K
    _shared['CH_K'] = CHANNEL_K

    metrics = defaultdict(list)
    n_recalled = 0
    n_total = len(eval_users)

    with Pool(RECALL_WORKERS) as pool:
        for uid, merged, actual_items in tqdm(
                pool.imap(_recall_one_user, eval_users, chunksize=256),
                total=n_total, desc="Recall+Rank"):

            if not set(actual_items) & set(merged):
                continue
            n_recalled += 1

            # Recall baseline metrics
            for ek in RANK_EVAL_KS:
                metrics[f'Recall_Baseline_HR@{ek}'].append(hitrate_at_k(actual_items, merged, k=ek))
                metrics[f'Recall_Baseline_NDCG@{ek}'].append(ndcg_at_k(actual_items, merged, k=ek))
            metrics['Recall_Baseline_MRR'].append(mrr(actual_items, merged))

            # Ranking reorder
            candidates = merged
            n_cand = len(candidates)
            cand_arr = np.array(candidates, dtype=np.int64)

            rank_features = {
                'user_id': torch.tensor([user_encoded_id[uid]] * n_cand, dtype=torch.long).to(device),
                'item_id': torch.from_numpy(item_encoded_id[cand_arr]).to(device),
                'user_top_genres': torch.from_numpy(np.tile(user_top_genres_arr[uid], (n_cand, 1))).to(device),
                'item_genres': torch.from_numpy(item_genres_arr[cand_arr]).to(device),
                'user_avg_rating': torch.tensor([user_avg_rating_arr[uid]] * n_cand, dtype=torch.float32).to(device),
                'user_activity': torch.tensor([user_activity_arr[uid]] * n_cand, dtype=torch.float32).to(device),
                'item_release_year': torch.from_numpy(item_release_year[cand_arr]).float().to(device),
                'item_avg_rating': torch.from_numpy(item_avg_rating[cand_arr]).float().to(device),
                'item_revenue': torch.from_numpy(item_revenue[cand_arr]).float().to(device),
                'item_budget': torch.from_numpy(item_budget[cand_arr]).float().to(device),
                'item_vote_count': torch.from_numpy(item_vote_count[cand_arr]).float().to(device),
            }

            with torch.no_grad():
                ctr_logit, pRating = ranking_model(rank_features)
                pCTR = torch.sigmoid(ctr_logit)
                pRating_clamped = pRating.clamp(min=0.01)
                final_score = (pCTR ** RANK_CTR_ALPHA) * (pRating_clamped ** RANK_RATING_BETA)
                order = final_score.cpu().numpy().argsort()[::-1]
                ranked = [candidates[i] for i in order]

            for ek in RANK_EVAL_KS:
                metrics[f'Ranking_HR@{ek}'].append(hitrate_at_k(actual_items, ranked, k=ek))
                metrics[f'Ranking_NDCG@{ek}'].append(ndcg_at_k(actual_items, ranked, k=ek))
            metrics['Ranking_MRR'].append(mrr(actual_items, ranked))

    _shared.clear()

    # Report
    print(f"\n=== Ranking Evaluation Report ===")
    print(f"Total users: {n_total}, Recalled target: {n_recalled} ({n_recalled/max(n_total,1)*100:.1f}%)\n")

    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    for name in sorted(metrics.keys()):
        vals = metrics[name]
        if vals:
            print(f"{name:<30} {np.mean(vals):>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Quick eval with 1000 users")
    args = parser.parse_args()
    evaluate(test_mode=args.test)
