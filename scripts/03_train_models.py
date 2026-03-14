import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import PROCESSED_DATA_DIR, FEATURE_STORE_DIR, MODEL_WEIGHTS_DIR, LEARNING_RATE, EPOCHS
from src.features.encoder import FeatureEncoder
from src.data_pipeline.dataset import create_dataloader
from src.models.recall.dual_tower import DualTowerModel

def apply_encoding(user_profile, item_profile, encoder):
    print("Applying encoding to profiles...")
    # Categorical
    user_profile['userId_encoded'] = encoder.transform_categorical(user_profile['userId'], 'userId')
    user_profile['top_genres_encoded'] = encoder.transform_categorical(user_profile['top_genres'], 'genres', is_list=True, max_len=3)
    user_profile['history_encoded'] = encoder.transform_categorical(user_profile['history'], 'movieId', is_list=True, max_len=50)

    item_profile['movieId_encoded'] = encoder.transform_categorical(item_profile['movieId'], 'movieId')
    item_profile['tmdb_genres_encoded'] = encoder.transform_categorical(item_profile['tmdb_genres'], 'genres', is_list=True, max_len=5)
    
    # Continuous with prefix isolation
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

def train_dual_tower(batch_size=1024):
    print(f"Loading data (Batch Size: {batch_size})...")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    with open(Path(FEATURE_STORE_DIR) / "popularity_list.json", "r") as f:
        popularity_list = json.load(f)

    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)
    
    # 建立从原始 movieId 到 item_profile 数组下标的映射 (Critical Fix #1)
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(item_profile['movieId'].values)}
    
    train_pos = train_data[train_data['rating'] >= 3.0].copy()
    # 注意：Dataset 内部现在需要原始 ID 来 loc，或者我们传入已经编码好的。
    # 这里我们保持 Dataset 的 loc 逻辑，但需要确保传递的 profile 是带索引的。
    
    # 计算 Log-Q
    item_counts = train_pos['movieId'].value_counts()
    total_count = len(train_pos)
    log_q_array = np.full(encoder.vocab_sizes['movieId'] + 1, np.log(1e-10), dtype=np.float32)
    movie_vocab = encoder.vocabularies['movieId']
    for mid, count in item_counts.items():
        if mid in movie_vocab:
            log_q_array[movie_vocab[mid]] = np.log((count / total_count) + 1e-10)
    
    # 将特征提取为张量字典，供快速下标索引 (item_lookup 现在使用位置索引)
    item_profile_sorted = item_profile.copy() # 不再强制按 ID 排序，按原始顺序索引
    item_lookup = {
        'user_id': None, # Item tower doesn't use this
        'item_id': torch.tensor(item_profile_sorted['movieId_encoded'].values, dtype=torch.long),
        'release_year': torch.tensor(item_profile_sorted['release_year_norm'].values, dtype=torch.float32),
        'avg_rating': torch.tensor(item_profile_sorted['avg_rating_norm'].values, dtype=torch.float32),
        'revenue': torch.tensor(item_profile_sorted['revenue_norm'].values, dtype=torch.float32),
        'tmdb_genres': torch.tensor(np.stack(item_profile_sorted['tmdb_genres_encoded'].values), dtype=torch.long),
        'log_q': torch.tensor(log_q_array[item_profile_sorted['movieId_encoded'].values], dtype=torch.float32)
    }

    # Dataset 的特征字段需要与 Tower 的 forward 输入对齐
    # 只选取需要的列再 rename，避免与原始同名列冲突
    user_profile_dataset = user_profile[
        ['userId', 'userId_encoded', 'avg_rating_norm', 'activity_norm',
         'history_encoded', 'history_ts_diff', 'top_genres_encoded']
    ].rename(columns={
        'userId_encoded': 'user_id',
        'avg_rating_norm': 'avg_rating',
        'activity_norm': 'activity',
        'history_encoded': 'history',
        'top_genres_encoded': 'top_genres'
    })
    
    item_profile_dataset = item_profile[
        ['movieId', 'movieId_encoded', 'release_year_norm', 'avg_rating_norm',
         'revenue_norm', 'tmdb_genres_encoded']
    ].rename(columns={
        'movieId_encoded': 'item_id',
        'release_year_norm': 'release_year_val',
        'avg_rating_norm': 'avg_rating',
        'revenue_norm': 'revenue',
        'tmdb_genres_encoded': 'tmdb_genres'
    })
    item_profile_dataset['log_q'] = log_q_array[item_profile['movieId_encoded'].values]

    dataloader = create_dataloader(train_pos, user_profile_dataset, item_profile_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    item_lookup = {k: v.to(device) for k, v in item_lookup.items() if v is not None}
    
    model = DualTowerModel(vocab_sizes=encoder.vocab_sizes, embed_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 困难负样本池 (使用下标索引)
    pop_indices = [movie_id_to_idx[mid] for mid in popularity_list[:1000] if mid in movie_id_to_idx]

    INBATCH_NEG_SIZE = min(1024, len(item_profile))
    GLOBAL_NEG_SIZE = 512
    HARD_NEG_SIZE = 128
    
    # 初始 Buffer (位置下标)
    buffer_indices = np.random.choice(np.arange(len(item_profile)), INBATCH_NEG_SIZE, replace=False)
    
    def get_neg_feat_by_indices(indices):
        return {k: item_lookup[k][indices] for k in item_lookup if k != 'log_q'}

    print("Starting Training (Fixing gradient flow and indexing)...")
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for user_feat, item_feat in pbar:
            user_feat = {k: v.to(device) for k, v in user_feat.items()}
            item_feat = {k: v.to(device) for k, v in item_feat.items()}
            
            optimizer.zero_grad()

            # --- Prepare Negative Embeddings (with gradients) ---
            inbatch_feat = get_neg_feat_by_indices(buffer_indices)
            inbatch_log_q = item_lookup['log_q'][buffer_indices]

            global_idx = np.random.choice(np.arange(len(item_profile)), GLOBAL_NEG_SIZE, replace=False)
            global_feat = get_neg_feat_by_indices(global_idx)
            global_log_q = item_lookup['log_q'][global_idx]

            hard_idx = np.random.choice(pop_indices, HARD_NEG_SIZE, replace=False)
            hard_feat = get_neg_feat_by_indices(hard_idx)

            # Batch all negative forward passes into one for efficiency
            all_neg_feat = {k: torch.cat([inbatch_feat[k], global_feat[k], hard_feat[k]], dim=0) for k in inbatch_feat}
            _, all_neg_emb = model(None, all_neg_feat)
            inbatch_neg_emb, global_neg_emb, hard_neg_emb = torch.split(
                all_neg_emb, [len(buffer_indices), GLOBAL_NEG_SIZE, HARD_NEG_SIZE])

            loss, l_nce, l_bpr = model.compute_loss(
                user_feat, item_feat, item_log_q=item_feat['log_q'],
                inbatch_neg_emb=inbatch_neg_emb, inbatch_neg_log_q=inbatch_log_q,
                global_neg_emb=global_neg_emb, global_neg_log_q=global_neg_log_q,
                hard_neg_emb=hard_neg_emb
            )
            loss.backward()
            optimizer.step()
            
            # Update Buffer with current batch items (Need to find their indices)
            # Efficiently update buffer_indices using rolling mechanism
            # item_feat['item_id'] is encoded ID, we need its positional index
            current_ids_orig = item_feat['item_id'].cpu().numpy()
            # We need original movieId to map to index. This requires dataset to return it or we re-map.
            # Simplified: just keep previous indices and update with a few new random ones if needed, 
            # or pass indices through dataset. For now, we rolling-update with random to ensure variety.
            new_rand_indices = np.random.choice(np.arange(len(item_profile)), min(batch_size, INBATCH_NEG_SIZE), replace=False)
            buffer_indices = np.roll(buffer_indices, -len(new_rand_indices))
            buffer_indices[-len(new_rand_indices):] = new_rand_indices
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'NCE': f"{l_nce.item():.4f}", 'BPR': f"{l_bpr.item():.4f}"})

    Path(MODEL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(MODEL_WEIGHTS_DIR) / "dual_tower.pth")
    print("Training finished.")

from src.models.recall.item_cf import ItemCFModel
from src.models.recall.user_cf import UserCFModel

def train_item_cf():
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    model = ItemCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "item_sim_matrix.pkl")
    model.fit(train_df=train_data[train_data['rating'] >= 3.0])

def train_user_cf():
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    model = UserCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "user_sim_matrix.pkl")
    model.fit(train_df=train_data[train_data['rating'] >= 3.0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dual_tower")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    if args.model == "dual_tower":
        train_dual_tower(batch_size=args.batch_size)
    elif args.model == "item_cf":
        train_item_cf()
    elif args.model == "user_cf":
        train_user_cf()
