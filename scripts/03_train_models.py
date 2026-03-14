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
from src.config.settings import PROCESSED_DATA_DIR, FEATURE_STORE_DIR, MODEL_WEIGHTS_DIR
from src.features.encoder import FeatureEncoder
from src.data_pipeline.dataset import create_dataloader
from src.models.recall.dual_tower import DualTowerModel

def apply_encoding(user_profile, item_profile, encoder):
    print("Applying encoding to profiles...")
    user_profile['userId_orig'] = user_profile['userId']
    user_profile['userId'] = encoder.transform_categorical(user_profile['userId'], 'userId')
    user_profile['top_genres'] = encoder.transform_categorical(user_profile['top_genres'], 'genres', is_list=True, max_len=3)
    user_profile['history'] = encoder.transform_categorical(user_profile['history'], 'movieId', is_list=True, max_len=50)

    item_profile['movieId_orig'] = item_profile['movieId']
    item_profile['movieId'] = encoder.transform_categorical(item_profile['movieId'], 'movieId')
    item_profile['tmdb_genres'] = encoder.transform_categorical(item_profile['tmdb_genres'], 'genres', is_list=True, max_len=5)
    
    user_cont = encoder.transform_continuous(user_profile, ['avg_rating', 'activity'])
    user_profile['avg_rating'] = user_cont['avg_rating']
    user_profile['activity'] = user_cont['activity']

    item_cont = encoder.transform_continuous(item_profile, ['release_year_orig', 'avg_rating', 'revenue', 'budget', 'vote_count_ml'])
    item_profile['release_year_val'] = item_cont['release_year_orig']
    item_profile['avg_rating'] = item_cont['avg_rating']
    item_profile['revenue'] = item_cont['revenue']
    item_profile['budget'] = item_cont['budget']
    item_profile['vote_count_ml'] = item_cont['vote_count_ml']
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
    
    movie_vocab = encoder.vocabularies['movieId']
    train_pos = train_data[train_data['rating'] >= 3.0].copy()
    train_pos['movieId'] = train_pos['movieId'].map(movie_vocab).fillna(0).astype(int)
    train_pos['userId'] = train_pos['userId'].map(encoder.vocabularies['userId']).fillna(0).astype(int)

    # Log-Q Calculation
    item_counts = train_pos['movieId'].value_counts()
    total_count = len(train_pos)
    log_q_array = np.full(encoder.vocab_sizes['movieId'] + 1, np.log(1e-10), dtype=np.float32)
    log_q_array[item_counts.index.values] = np.log((item_counts.values / total_count) + 1e-10)
    item_profile['log_q'] = log_q_array[item_profile['movieId'].values]

    # Hard Neg Pool (Top 1000)
    encoded_pop_list = [movie_vocab[mid] for mid in popularity_list[:1000] if mid in movie_vocab]

    item_profile_sorted = item_profile.sort_values('movieId')
    item_lookup = {
        'item_id': torch.tensor(item_profile_sorted['movieId'].values, dtype=torch.long),
        'release_year': torch.tensor(item_profile_sorted['release_year_val'].values, dtype=torch.float32),
        'avg_rating': torch.tensor(item_profile_sorted['avg_rating'].values, dtype=torch.float32),
        'revenue': torch.tensor(item_profile_sorted['revenue'].values, dtype=torch.float32),
        'tmdb_genres': torch.tensor(np.stack(item_profile_sorted['tmdb_genres'].values), dtype=torch.long),
        'log_q': torch.tensor(item_profile_sorted['log_q'].values, dtype=torch.float32)
    }

    dataloader = create_dataloader(train_pos, user_profile, item_profile, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    item_lookup = {k: v.to(device) for k, v in item_lookup.items()}
    
    model = DualTowerModel(vocab_sizes=encoder.vocab_sizes, embed_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Fixed Size Buffers ---
    INBATCH_NEG_SIZE = 1024
    GLOBAL_NEG_SIZE = 512
    HARD_NEG_SIZE = 128
    
    # Initialize in-batch buffer with random items
    buffer_indices = np.random.choice(np.arange(len(item_profile)), INBATCH_NEG_SIZE, replace=False)
    
    print(f"Starting Training. Fixed Negs: {INBATCH_NEG_SIZE} in-batch pool, {GLOBAL_NEG_SIZE} global, {HARD_NEG_SIZE} hard.")
    for epoch in range(3):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for user_feat, item_feat in pbar:
            user_feat = {k: v.to(device) for k, v in user_feat.items()}
            item_feat = {k: v.to(device) for k, v in item_feat.items()}
            
            # 1. Prepare Negative Embeddings (No-grad)
            with torch.no_grad():
                # In-batch pool
                inbatch_neg_feat = {k: item_lookup[k][buffer_indices] for k in item_lookup if k != 'log_q'}
                inbatch_neg_log_q = item_lookup['log_q'][buffer_indices]
                _, inbatch_neg_emb = model(None, inbatch_neg_feat)
                
                # Global random
                global_idx = np.random.choice(np.arange(len(item_profile)), GLOBAL_NEG_SIZE, replace=True)
                global_neg_feat = {k: item_lookup[k][global_idx] for k in item_lookup if k != 'log_q'}
                global_neg_log_q = item_lookup['log_q'][global_idx]
                _, global_neg_emb = model(None, global_neg_feat)
                
                # Hard popular
                hard_idx = np.random.choice(encoded_pop_list, HARD_NEG_SIZE, replace=True)
                hard_neg_feat = {k: item_lookup[k][hard_idx] for k in item_lookup if k != 'log_q'}
                _, hard_neg_emb = model(None, hard_neg_feat)

            # 2. Main Step
            optimizer.zero_grad()
            loss, l_nce, l_bpr = model.compute_loss(
                user_feat, item_feat, item_log_q=item_feat['log_q'],
                inbatch_neg_emb=inbatch_neg_emb, inbatch_neg_log_q=inbatch_neg_log_q,
                global_neg_emb=global_neg_emb, global_neg_log_q=global_neg_log_q,
                hard_neg_emb=hard_neg_emb
            )
            loss.backward()
            optimizer.step()
            
            # 3. Update Buffer (FIFO)
            # Use current batch items to update the pool for NEXT step
            new_ids = item_feat['item_id'].cpu().numpy()
            if len(new_ids) >= INBATCH_NEG_SIZE:
                buffer_indices = new_ids[:INBATCH_NEG_SIZE]
            else:
                # Rolling update
                num_new = len(new_ids)
                buffer_indices = np.roll(buffer_indices, -num_new)
                buffer_indices[-num_new:] = new_ids
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'NCE': f"{l_nce.item():.4f}"})

    torch.save(model.state_dict(), Path(MODEL_WEIGHTS_DIR) / "dual_tower.pth")
    print("Training finished.")

# Helper functions for ItemCF/UserCF
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
