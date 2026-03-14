import argparse
import sys
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

    print("Calculating Log-Q (Sampling Probabilities) for Correction...")
    # Count occurrences of each movieId in training data
    item_counts = train_data['movieId'].value_counts()
    # Map back to encoded indices
    # We use a small smoothing factor to avoid log(0)
    total_count = len(train_data)
    
    def get_log_q(row):
        # We need the original movieId (unencoded) or we map our counts to the new indices
        # Let's use the encoder's vocabulary to map correctly
        orig_id = row.name # This is index if we're not careful. 
        # Actually, simpler: map the counts directly to item_profile.
        return np.log((item_counts.get(row['movieId_orig'], 0) / total_count) + 1e-10)

    # Temporary store original ID to match counts
    # item_profile['movieId'] is already encoded. We need to match with train_data's movieId.
    # Let's map counts to encoded IDs.
    movie_vocab = encoder.vocabularies['movieId']
    encoded_counts = {}
    for orig_id, count in item_counts.items():
        if orig_id in movie_vocab:
            encoded_counts[movie_vocab[orig_id]] = count
    
    def calculate_item_log_q(encoded_id):
        cnt = encoded_counts.get(encoded_id, 0)
        return np.log((cnt / total_count) + 1e-10)

    item_profile['log_q'] = item_profile['movieId'].apply(calculate_item_log_q)

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

    print("Starting Training (with Log-Q Correction)...")
    epochs = 3
    num_neg_per_batch = 512
    all_item_indices = np.arange(len(item_profile))

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for user_feat, item_feat in pbar:
            user_feat = {k: v.to(device) for k, v in user_feat.items()}
            item_feat = {k: v.to(device) for k, v in item_feat.items()}
            
            # Sampling global negatives
            neg_indices = np.random.choice(all_item_indices, num_neg_per_batch, replace=True)
            neg_feat = {
                'item_id': item_lookup['item_id'][neg_indices],
                'release_year': item_lookup['release_year'][neg_indices],
                'avg_rating': item_lookup['avg_rating'][neg_indices],
                'revenue': item_lookup['revenue'][neg_indices],
                'tmdb_genres': item_lookup['tmdb_genres'][neg_indices]
            }
            neg_log_q = item_lookup['log_q'][neg_indices]
            
            optimizer.zero_grad()
            # Pass log_q of positive items and negative items for correction
            loss = model.compute_loss(
                user_feat, 
                item_feat, 
                item_log_q=item_feat['log_q'], 
                extra_item_features=neg_feat, 
                extra_item_log_q=neg_log_q
            )
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

    Path(MODEL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(MODEL_WEIGHTS_DIR) / "dual_tower.pth")
    print("Model saved.")

if __name__ == "__main__":
    train_dual_tower()
