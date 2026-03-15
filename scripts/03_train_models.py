import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import (
    PROCESSED_DATA_DIR, FEATURE_STORE_DIR, MODEL_WEIGHTS_DIR,
    EMBEDDING_DIM, TAU, TIME_DECAY_LAMBDA, BPR_GAMMA, BPR_MARGIN,
    LOSS_INFONCE_WEIGHT, LOSS_BPR_WEIGHT, LOGIT_SCALE_MAX, CONT_BUCKET_SIZE,
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    INBATCH_NEG_SIZE, GLOBAL_NEG_SIZE, HARD_NEG_SIZE,
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN
)
from src.features.encoder import FeatureEncoder
from src.data_pipeline.dataset import create_dataloader
from src.models.recall.dual_tower import DualTowerModel
from src.models.recall.item_cf import ItemCFModel
from src.models.recall.user_cf import UserCFModel

def apply_encoding(user_profile, item_profile, encoder):
    print("Applying encoding to profiles...")
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

def train_dual_tower(batch_size=BATCH_SIZE):
    print(f"=== Training Dual-Tower Model (Batch: {batch_size}, Epochs: {EPOCHS}) ===")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    with open(Path(FEATURE_STORE_DIR) / "popularity_list.json", "r") as f:
        popularity_list = json.load(f)

    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)
    
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(item_profile['movieId'].values)}
    train_pos = train_data[train_data['rating'] >= 3.0].copy()
    
    item_counts = train_pos['movieId'].value_counts()
    total_count = len(train_pos)
    log_q_array = np.full(encoder.vocab_sizes['movieId'] + 1, np.log(1e-10), dtype=np.float32)
    movie_vocab = encoder.vocabularies['movieId']
    for mid, count in item_counts.items():
        if mid in movie_vocab:
            log_q_array[movie_vocab[mid]] = np.log((count / total_count) + 1e-10)
    
    item_lookup = {
        'item_id': torch.tensor(item_profile['movieId_encoded'].values, dtype=torch.long),
        'release_year': torch.tensor(item_profile['release_year_norm'].values, dtype=torch.float32),
        'avg_rating': torch.tensor(item_profile['avg_rating_norm'].values, dtype=torch.float32),
        'revenue': torch.tensor(item_profile['revenue_norm'].values, dtype=torch.float32),
        'tmdb_genres': torch.tensor(np.stack(item_profile['tmdb_genres_encoded'].values), dtype=torch.long),
        'log_q': torch.tensor(log_q_array[item_profile['movieId_encoded'].values], dtype=torch.float32)
    }

    user_profile_dataset = user_profile[['userId', 'userId_encoded', 'avg_rating_norm', 'activity_norm', 'history_encoded', 'history_ts_diff', 'top_genres_encoded']].rename(columns={
        'userId_encoded': 'user_id', 'avg_rating_norm': 'avg_rating', 'activity_norm': 'activity', 'history_encoded': 'history', 'top_genres_encoded': 'top_genres'
    })
    
    item_profile_dataset = item_profile[['movieId', 'movieId_encoded', 'release_year_norm', 'avg_rating_norm', 'revenue_norm', 'tmdb_genres_encoded']].rename(columns={
        'movieId_encoded': 'item_id', 'release_year_norm': 'release_year_val', 'avg_rating_norm': 'avg_rating', 'revenue_norm': 'revenue', 'tmdb_genres_encoded': 'tmdb_genres'
    })
    item_profile_dataset['log_q'] = log_q_array[item_profile['movieId_encoded'].values]

    dataloader = create_dataloader(train_pos, user_profile_dataset, item_profile_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    item_lookup = {k: v.to(device) for k, v in item_lookup.items()}

    # Compute quantile bucket boundaries from training profiles
    def _quantile_bounds(values, n_buckets):
        quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]
        return np.quantile(values[~np.isnan(values)], quantiles).astype(np.float32)

    user_bucket_bounds = {
        'avg_rating': _quantile_bounds(user_profile['avg_rating_norm'].values, CONT_BUCKET_SIZE),
        'activity': _quantile_bounds(user_profile['activity_norm'].values, CONT_BUCKET_SIZE),
    }
    item_bucket_bounds = {
        'release_year': _quantile_bounds(item_profile['release_year_norm'].values, CONT_BUCKET_SIZE),
        'avg_rating': _quantile_bounds(item_profile['avg_rating_norm'].values, CONT_BUCKET_SIZE),
        'revenue': _quantile_bounds(item_profile['revenue_norm'].values, CONT_BUCKET_SIZE),
    }

    model = DualTowerModel(
        vocab_sizes=encoder.vocab_sizes, embed_dim=EMBEDDING_DIM, tau=TAU,
        time_decay_lambda=TIME_DECAY_LAMBDA, bpr_gamma=BPR_GAMMA, bpr_margin=BPR_MARGIN,
        loss_infonce_weight=LOSS_INFONCE_WEIGHT, loss_bpr_weight=LOSS_BPR_WEIGHT,
        logit_scale_max=LOGIT_SCALE_MAX, cont_bucket_size=CONT_BUCKET_SIZE,
        user_bucket_boundaries=user_bucket_bounds, item_bucket_boundaries=item_bucket_bounds
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * EPOCHS, eta_min=1e-6)

    pop_indices = [movie_id_to_idx[mid] for mid in popularity_list[:5000] if mid in movie_id_to_idx]
    pop_indices_t = torch.tensor(pop_indices, dtype=torch.long, device=device)
    n_items = len(item_profile)
    buffer_indices = torch.randperm(n_items, device=device)[:INBATCH_NEG_SIZE]

    def get_neg_feat_by_indices(indices):
        return {k: item_lookup[k][indices] for k in item_lookup if k != 'log_q'}

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for user_feat, item_feat in pbar:
            user_feat = {k: v.to(device, non_blocking=True) for k, v in user_feat.items()}
            item_feat = {k: v.to(device, non_blocking=True) for k, v in item_feat.items()}
            optimizer.zero_grad()

            inbatch_log_q = item_lookup['log_q'][buffer_indices]
            global_idx = torch.randint(n_items, (GLOBAL_NEG_SIZE,), device=device)
            global_log_q = item_lookup['log_q'][global_idx]
            hard_idx = pop_indices_t[torch.randint(len(pop_indices_t), (HARD_NEG_SIZE,), device=device)]
            
            all_neg_indices = torch.cat([buffer_indices, global_idx])
            all_neg_ids = item_lookup['item_id'][all_neg_indices]
            user_hist_ids = user_feat['history']
            hard_neg_ids = item_lookup['item_id'][hard_idx]

            infonce_collision = (all_neg_ids.unsqueeze(0).unsqueeze(-1) == user_hist_ids.unsqueeze(1)).any(dim=-1)
            bpr_collision = (hard_neg_ids.unsqueeze(0).unsqueeze(-1) == user_hist_ids.unsqueeze(1)).any(dim=-1)
            
            neg_feat = get_neg_feat_by_indices(all_neg_indices)
            hard_feat = get_neg_feat_by_indices(hard_idx)
            total_neg_feat = {k: torch.cat([neg_feat[k], hard_feat[k]]) for k in neg_feat}
            _, total_neg_emb = model(None, total_neg_feat)
            
            neg_emb, hard_neg_emb = torch.split(total_neg_emb, [len(all_neg_indices), HARD_NEG_SIZE])
            inbatch_neg_emb, global_neg_emb = torch.split(neg_emb, [len(buffer_indices), GLOBAL_NEG_SIZE])

            loss, l_nce, l_bpr = model.compute_loss(
                user_feat, item_feat, item_log_q=item_feat['log_q'],
                inbatch_neg_emb=inbatch_neg_emb, inbatch_neg_log_q=inbatch_log_q,
                global_neg_emb=global_neg_emb, global_neg_log_q=global_log_q,
                hard_neg_emb=hard_neg_emb,
                collision_mask=infonce_collision,
                bpr_collision_mask=bpr_collision
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            new_indices = torch.randint(n_items, (min(batch_size, INBATCH_NEG_SIZE),), device=device)
            buffer_indices = torch.cat([buffer_indices[len(new_indices):], new_indices])
            tau_eff = 1.0 / model.logit_scale.exp().clamp(max=model.logit_scale_max).item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'NCE': f"{l_nce.item():.4f}", 'BPR': f"{l_bpr.item():.4f}", 'tau': f"{tau_eff:.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.6f}"})

    Path(MODEL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(MODEL_WEIGHTS_DIR) / "dual_tower.pth")
    print("Dual-Tower training finished.")

def train_item_cf():
    print("=== Training ItemCF (Sparse Matrix) ===")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    model = ItemCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "item_sim_matrix.pkl")
    model.fit(train_df=train_data[train_data['rating'] >= 3.0])
    print("ItemCF training finished.")

def train_user_cf():
    print("=== Training UserCF (Sparse Matrix) ===")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    model = UserCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "user_sim_matrix.pkl")
    model.fit(train_df=train_data[train_data['rating'] >= 3.0])
    print("UserCF training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["dual_tower", "item_cf", "user_cf", "all"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    
    if args.model == "dual_tower" or args.model == "all":
        train_dual_tower(batch_size=args.batch_size)
    if args.model == "item_cf" or args.model == "all":
        train_item_cf()
    if args.model == "user_cf" or args.model == "all":
        train_user_cf()
