import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import (
    PROCESSED_DATA_DIR, FEATURE_STORE_DIR, MODEL_WEIGHTS_DIR,
    EMBEDDING_DIM, TAU, TIME_DECAY_LAMBDA, BPR_GAMMA,
    LOSS_INFONCE_WEIGHT, LOSS_BPR_WEIGHT, RECALL_K,
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN,
    MERGER_WEIGHTS
)
from src.features.encoder import FeatureEncoder
from src.models.recall.dual_tower import DualTowerModel
from src.models.recall.item_cf import ItemCFModel
from src.models.recall.user_cf import UserCFModel
from src.models.recall.simple_recall import PopularityRecall, GenreRecall
from src.models.recall.merger import RecallMerger
from src.evaluation.metrics import recall_at_k, ndcg_at_k

def apply_encoding(user_profile, item_profile, encoder):
    """Reuse the encoding logic from training."""
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
    return user_profile, item_profile

def pad_ts_diff(ts_list, max_len):
    if not isinstance(ts_list, (list, np.ndarray)):
        return [0.0] * max_len
    ts = list(ts_list)[:max_len]
    return ts + [0.0] * (max_len - len(ts))

def evaluate(test_mode=False):
    print("=== Starting End-to-End Evaluation ===")

    CHANNEL_K = 200       # 每个通道召回的候选数
    MERGE_K = 500         # 融合后保留的候选数
    CHANNEL_EVAL_KS = [50, 100, 200]   # 单通道评测的 K 值
    FINAL_EVAL_KS = [100, 200, 500]    # 融合后评测的 K 值

    # 1. Load Data
    val_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "val_data.parquet")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")

    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    # 2. Load Models
    model = DualTowerModel(
        vocab_sizes=encoder.vocab_sizes, embed_dim=EMBEDDING_DIM, tau=TAU,
        time_decay_lambda=TIME_DECAY_LAMBDA, bpr_gamma=BPR_GAMMA,
        loss_infonce_weight=LOSS_INFONCE_WEIGHT, loss_bpr_weight=LOSS_BPR_WEIGHT
    ).to(device)
    model_path = Path(MODEL_WEIGHTS_DIR) / "dual_tower.pth"
    dual_tower_ready = False
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        dual_tower_ready = True
        print("Loaded Dual-Tower weights.")
    model.eval()

    item_cf = ItemCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "item_sim_matrix.pkl")
    item_cf_ready = False
    try:
        item_cf.load()
        item_cf_ready = True
    except FileNotFoundError:
        print("Warning: ItemCF matrix not found.")

    user_cf = UserCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "user_sim_matrix.pkl")
    user_cf_ready = False
    try:
        user_cf.load()
        user_cf_ready = True
    except FileNotFoundError:
        print("Warning: UserCF matrix not found.")

    pop_recall = PopularityRecall(item_profile_path=Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    genre_recall = GenreRecall(
        genre_to_items_path=Path(FEATURE_STORE_DIR) / "genre_to_items.json",
        item_profile_path=Path(PROCESSED_DATA_DIR) / "item_profile.parquet"
    )
    merger = RecallMerger(top_k=MERGE_K)

    # 3. Build FAISS Index
    idx_to_movie_id = []
    index = None
    if dual_tower_ready:
        print("Building FAISS index for Vector Recall...")
        item_profile_sorted = item_profile.sort_values('movieId_encoded')
        item_feat_tensors = {
            'item_id': torch.tensor(item_profile_sorted['movieId_encoded'].values, dtype=torch.long).to(device),
            'release_year': torch.tensor(item_profile_sorted['release_year_norm'].values, dtype=torch.float32).to(device),
            'avg_rating': torch.tensor(item_profile_sorted['avg_rating_norm'].values, dtype=torch.float32).to(device),
            'revenue': torch.tensor(item_profile_sorted['revenue_norm'].values, dtype=torch.float32).to(device),
            'tmdb_genres': torch.tensor(np.stack(item_profile_sorted['tmdb_genres_encoded'].values), dtype=torch.long).to(device)
        }
        with torch.no_grad():
            _, all_item_embs = model(None, item_feat_tensors)
            all_item_embs = all_item_embs.cpu().numpy().astype('float32')

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(all_item_embs)
        idx_to_movie_id = item_profile_sorted['movieId'].values
        print(f"FAISS index built: {index.ntotal} items.")

    # 4. Prepare Eval Users
    val_ground_truth = val_data.groupby('userId')['movieId'].apply(list).to_dict()
    user_profile_indexed = user_profile.set_index('userId')
    eval_users = list(val_ground_truth.keys())
    if test_mode:
        eval_users = eval_users[:1000]

    # 5. Evaluate
    print(f"Evaluating {len(eval_users)} users...")
    metrics = defaultdict(list)

    for uid in tqdm(eval_users, desc="Evaluating"):
        if uid not in user_profile_indexed.index:
            continue
        user_row = user_profile_indexed.loc[uid]
        actual_items = val_ground_truth[uid]
        channels = {}

        # Dual Tower
        if dual_tower_ready:
            ts_padded = pad_ts_diff(user_row['history_ts_diff'], USER_HISTORY_MAX_LEN)
            user_tensor = {
                'user_id': torch.tensor([user_row['userId_encoded']], dtype=torch.long).to(device),
                'avg_rating': torch.tensor([user_row['avg_rating_norm']], dtype=torch.float32).to(device),
                'activity': torch.tensor([user_row['activity_norm']], dtype=torch.float32).to(device),
                'history': torch.tensor([user_row['history_encoded']], dtype=torch.long).to(device),
                'history_ts_diff': torch.tensor([ts_padded], dtype=torch.float32).to(device),
                'top_genres': torch.tensor([user_row['top_genres_encoded']], dtype=torch.long).to(device)
            }
            with torch.no_grad():
                u_emb, _ = model(user_tensor, None)
                u_emb = u_emb.cpu().numpy().astype('float32')
            _, I = index.search(u_emb, CHANNEL_K)
            channels['dual_tower'] = [int(idx_to_movie_id[i]) for i in I[0]]
            for ek in CHANNEL_EVAL_KS:
                metrics[f'Recall@{ek}_DualTower'].append(recall_at_k(actual_items, channels['dual_tower'], k=ek))

        # ItemCF
        if item_cf_ready:
            history_orig = user_row.get('history')
            channels['item_cf'] = item_cf.retrieve(list(history_orig), k=CHANNEL_K) if history_orig is not None else []
            for ek in CHANNEL_EVAL_KS:
                metrics[f'Recall@{ek}_ItemCF'].append(recall_at_k(actual_items, channels['item_cf'], k=ek))

        # UserCF
        if user_cf_ready:
            channels['user_cf'] = user_cf.retrieve(uid, k=CHANNEL_K)
            for ek in CHANNEL_EVAL_KS:
                metrics[f'Recall@{ek}_UserCF'].append(recall_at_k(actual_items, channels['user_cf'], k=ek))

        # Pop
        channels['popularity'] = pop_recall.retrieve(k=CHANNEL_K)
        for ek in CHANNEL_EVAL_KS:
            metrics[f'Recall@{ek}_Pop'].append(recall_at_k(actual_items, channels['popularity'], k=ek))

        # Genre
        top_genres = user_row.get('top_genres')
        if isinstance(top_genres, (list, np.ndarray)):
            channels['genre'] = genre_recall.retrieve(list(top_genres), k=CHANNEL_K)
            for ek in CHANNEL_EVAL_KS:
                metrics[f'Recall@{ek}_Genre'].append(recall_at_k(actual_items, channels['genre'], k=ek))

        # 5.1 Filter Watched Items (Critical Optimization)
        # Get set of original movieIds the user has already seen
        watched_set = set(user_row['history']) if isinstance(user_row['history'], (list, np.ndarray)) else set()
        
        for name in list(channels.keys()):
            # Filter out watched items from each channel's candidates
            filtered_list = [iid for iid in channels[name] if iid not in watched_set]
            channels[name] = filtered_list[:CHANNEL_K]

        # 5.2 Merge with optimized weights
        merged = merger.merge(channels, weights=MERGER_WEIGHTS)
        for ek in FINAL_EVAL_KS:
            metrics[f'Recall@{ek}_Final'].append(recall_at_k(actual_items, merged, k=ek))
            metrics[f'NDCG@{ek}_Final'].append(ndcg_at_k(actual_items, merged, k=ek))

    # 6. Report
    print("\n=== Multi-Channel Metrics Report ===")
    for name, vals in metrics.items():
        if vals:
            print(f"{name:25}: {np.mean(vals):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Quick evaluation with 1000 users.")
    args = parser.parse_args()
    evaluate(test_mode=args.test)
