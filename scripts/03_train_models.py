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

def train_dual_tower(batch_size=1024, neg_pool_size=8192):
    print(f"Loading data (Batch Size: {batch_size}, Neg Pool: {neg_pool_size})...")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    
    with open(Path(FEATURE_STORE_DIR) / "popularity_list.json", "r") as f:
        popularity_list = json.load(f)

    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()

    # Map movieId/userId in train_data to encoded indices
    movie_vocab = encoder.vocabularies['movieId']
    user_vocab = encoder.vocabularies['userId']
    
    # Pre-encoding profiles
    from scripts.03_train_models import apply_encoding # Reuse the logic
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)
    
    train_pos = train_data[train_data['rating'] >= 3.0].copy()
    train_pos['movieId'] = train_pos['movieId'].map(movie_vocab).fillna(0).astype(int)
    train_pos['userId'] = train_pos['userId'].map(user_vocab).fillna(0).astype(int)

    # Log-Q Calculation
    item_counts = train_pos['movieId'].value_counts()
    total_count = len(train_pos)
    log_q_array = np.full(encoder.vocab_sizes['movieId'] + 1, np.log(1e-10), dtype=np.float32)
    log_q_array[item_counts.index.values] = np.log((item_counts.values / total_count) + 1e-10)
    item_profile['log_q'] = log_q_array[item_profile['movieId'].values]

    # Hard Negative Pool
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
    print(f"Using device: {device}")

    item_lookup = {k: v.to(device) for k, v in item_lookup.items()}
    model = DualTowerModel(vocab_sizes=encoder.vocab_sizes, embed_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- FIFO Queue for Large Negative Pool ---
    # We store embeddings and their corresponding log_q
    embed_dim = 64
    neg_queue_emb = torch.randn(neg_pool_size, embed_dim).to(device)
    neg_queue_log_q = torch.full((neg_pool_size,), np.log(1e-10)).to(device)
    queue_ptr = 0

    print("Starting Training with Large Negative Pool Cache...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for user_feat, item_feat in pbar:
            user_feat = {k: v.to(device) for k, v in user_feat.items()}
            item_feat = {k: v.to(device) for k, v in item_feat.items()}
            
            # Simple Global Random Negs
            simple_idx = np.random.choice(np.arange(len(item_profile)), 256, replace=True)
            simple_feat = {k: v[simple_idx] for k, v in item_lookup.items() if k != 'log_q'}
            simple_log_q = item_lookup['log_q'][simple_idx]
            
            # Hard Negs
            hard_idx = np.random.choice(encoded_pop_list, 128, replace=True)
            hard_feat = {k: v[hard_idx] for k, v in item_lookup.items() if k != 'log_q'}
            
            optimizer.zero_grad()
            
            # Forward pass to get current embeddings for queue update
            with torch.no_grad():
                _, current_item_emb = model(user_feat, item_feat)
            
            loss, l_nce, l_bpr = model.compute_loss(
                user_feat, item_feat, item_log_q=item_feat['log_q'], 
                simple_neg_features=simple_feat, simple_neg_log_q=simple_log_q,
                hard_neg_features=hard_feat,
                cached_item_emb=neg_queue_emb, cached_item_log_q=neg_queue_log_q
            )
            
            loss.backward()
            optimizer.step()
            
            # --- Update FIFO Queue ---
            batch_size_actual = current_item_emb.shape[0]
            if queue_ptr + batch_size_actual > neg_pool_size:
                # Wrap around
                space_left = neg_pool_size - queue_ptr
                neg_queue_emb[queue_ptr:] = current_item_emb[:space_left].detach()
                neg_queue_log_q[queue_ptr:] = item_feat['log_q'][:space_left].detach()
                queue_ptr = 0
            else:
                neg_queue_emb[queue_ptr:queue_ptr+batch_size_actual] = current_item_emb.detach()
                neg_queue_log_q[queue_ptr:queue_ptr+batch_size_actual] = item_feat['log_q'].detach()
                queue_ptr += batch_size_actual
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'NCE': f"{l_nce.item():.4f}"})

    torch.save(model.state_dict(), Path(MODEL_WEIGHTS_DIR) / "dual_tower.pth")
    print("Training finished.")

# Import necessary functions for re-runability if this is a standalone script
from scripts.03_train_models import apply_encoding, train_item_cf, train_user_cf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dual_tower")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--neg_pool", type=int, default=8192)
    args = parser.parse_args()
    
    if args.model == "dual_tower":
        train_dual_tower(batch_size=args.batch_size, neg_pool_size=args.neg_pool)
    elif args.model == "item_cf":
        train_item_cf()
    elif args.model == "user_cf":
        train_user_cf()
