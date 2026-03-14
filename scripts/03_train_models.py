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

def train_dual_tower():
    print("Loading data...")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    
    with open(Path(FEATURE_STORE_DIR) / "popularity_list.json", "r") as f:
        popularity_list = json.load(f)

    # Positive samples for training
    train_data = train_data[train_data['rating'] >= 3.0]

    print("Encoding features...")
    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.fit_categorical(user_profile['userId'], 'userId')
    encoder.fit_categorical(item_profile['movieId'], 'movieId')
    all_genres = pd.concat([user_profile['top_genres'].explode(), item_profile['tmdb_genres'].explode()]).dropna()
    encoder.fit_categorical(all_genres, 'genres')

    user_profile['userId'] = encoder.transform_categorical(user_profile['userId'], 'userId')
    user_profile['top_genres'] = encoder.transform_categorical(user_profile['top_genres'], 'genres', is_list=True, max_len=3)
    user_profile['history'] = encoder.transform_categorical(user_profile['history'], 'movieId', is_list=True, max_len=50)

    item_profile['movieId'] = encoder.transform_categorical(item_profile['movieId'], 'movieId')
    item_profile['tmdb_genres'] = encoder.transform_categorical(item_profile['tmdb_genres'], 'genres', is_list=True, max_len=5)

    encoder.fit_continuous(user_profile, ['avg_rating', 'activity'])
    user_cont = encoder.transform_continuous(user_profile, ['avg_rating', 'activity'])
    user_profile['avg_rating'] = user_cont['avg_rating']
    user_profile['activity'] = user_cont['activity']

    encoder.fit_continuous(item_profile, ['release_year', 'avg_rating', 'revenue'])
    item_cont = encoder.transform_continuous(item_profile, ['release_year', 'avg_rating', 'revenue'])
    item_profile['release_year'] = item_cont['release_year']
    item_profile['avg_rating'] = item_cont['avg_rating']
    item_profile['revenue'] = item_cont['revenue']
    encoder.save()

    print("Calculating Log-Q (Sampling Probabilities)...")
    item_counts = train_data['movieId'].value_counts()
    total_count = len(train_data)
    max_id = encoder.vocab_sizes['movieId']
    log_q_array = np.full(max_id + 1, np.log(1e-10), dtype=np.float32)
    movie_vocab = encoder.vocabularies['movieId']
    
    encoded_ids = [movie_vocab[orig_id] for orig_id in item_counts.index if orig_id in movie_vocab]
    counts = [item_counts[orig_id] for orig_id in item_counts.index if orig_id in movie_vocab]
    if encoded_ids:
        log_q_array[encoded_ids] = np.log((np.array(counts) / total_count) + 1e-10)
    item_profile['log_q'] = log_q_array[item_profile['movieId'].values]

    # Pre-encode Top 1000 items as potential hard negatives (Popular pool)
    print("Pre-encoding Popularity List for Hard Negatives...")
    encoded_pop_list = [movie_vocab[mid] for mid in popularity_list[:1000] if mid in movie_vocab]

    print("Preparing Global Item Features Lookup...")
    item_profile = item_profile.sort_values('movieId')
    item_lookup = {
        'item_id': torch.tensor(item_profile['movieId'].values, dtype=torch.long),
        'release_year': torch.tensor(item_profile['release_year'].values, dtype=torch.float32),
        'avg_rating': torch.tensor(item_profile['avg_rating'].values, dtype=torch.float32),
        'revenue': torch.tensor(item_profile['revenue'].values, dtype=torch.float32),
        'tmdb_genres': torch.tensor(np.stack(item_profile['tmdb_genres'].values), dtype=torch.long),
        'log_q': torch.tensor(item_profile['log_q'].values, dtype=torch.float32)
    }

    print("Building DataLoader...")
    dataloader = create_dataloader(train_data, user_profile, item_profile, batch_size=1024)

    print("Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    item_lookup = {k: v.to(device) for k, v in item_lookup.items()}
    model = DualTowerModel(vocab_sizes=encoder.vocab_sizes, embed_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting Training (InfoNCE for Simple Negatives, BPR for Hard Negatives)...")
    epochs = 3
    num_simple_neg = 512
    num_hard_neg = 128
    all_item_indices = np.arange(len(item_profile))

    def get_batch_features(indices):
        return {
            'item_id': item_lookup['item_id'][indices],
            'release_year': item_lookup['release_year'][indices],
            'avg_rating': item_lookup['avg_rating'][indices],
            'revenue': item_lookup['revenue'][indices],
            'tmdb_genres': item_lookup['tmdb_genres'][indices]
        }

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for user_feat, item_feat in pbar:
            user_feat = {k: v.to(device) for k, v in user_feat.items()}
            item_feat = {k: v.to(device) for k, v in item_feat.items()}
            
            # 1. Sample Simple Negatives (Global Random)
            simple_neg_idx = np.random.choice(all_item_indices, num_simple_neg, replace=True)
            simple_neg_feat = get_batch_features(simple_neg_idx)
            simple_neg_log_q = item_lookup['log_q'][simple_neg_idx]
            
            # 2. Sample Hard Negatives (Popular)
            hard_neg_idx = np.random.choice(encoded_pop_list, num_hard_neg, replace=True)
            hard_neg_feat = get_batch_features(hard_neg_idx)
            
            optimizer.zero_grad()
            # Dual-Tower Mixed Loss
            loss, l_infonce, l_bpr = model.compute_loss(
                user_feat, 
                item_feat, 
                item_log_q=item_feat['log_q'], 
                simple_neg_features=simple_neg_feat,
                simple_neg_log_q=simple_neg_log_q,
                hard_neg_features=hard_neg_feat
            )
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'NCE': f"{l_infonce.item():.4f}", 'BPR': f"{l_bpr.item():.4f}"})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

    Path(MODEL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(MODEL_WEIGHTS_DIR) / "dual_tower.pth")
    print("Model saved.")

if __name__ == "__main__":
    train_dual_tower()
