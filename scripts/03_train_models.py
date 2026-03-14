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
from src.models.recall.item_cf import ItemCFModel
from src.models.recall.user_cf import UserCFModel

def apply_encoding(user_profile, item_profile, encoder):
    """根据加载的 encoder 对 profile 进行转换"""
    print("Applying encoding to profiles...")
    # 类别特征
    user_profile['userId_orig'] = user_profile['userId']
    user_profile['userId'] = encoder.transform_categorical(user_profile['userId'], 'userId')
    user_profile['top_genres'] = encoder.transform_categorical(user_profile['top_genres'], 'genres', is_list=True, max_len=3)
    user_profile['history'] = encoder.transform_categorical(user_profile['history'], 'movieId', is_list=True, max_len=50)

    item_profile['movieId_orig'] = item_profile['movieId']
    item_profile['movieId'] = encoder.transform_categorical(item_profile['movieId'], 'movieId')
    item_profile['tmdb_genres'] = encoder.transform_categorical(item_profile['tmdb_genres'], 'genres', is_list=True, max_len=5)
    item_profile['release_year_encoded'] = encoder.transform_categorical(item_profile['release_year'], 'genres', is_list=False) # Reuse genres vocab for simple strings if needed or just use LabelEncoder

    # 连续特征
    user_cont = encoder.transform_continuous(user_profile, ['avg_rating', 'activity'])
    user_profile['avg_rating'] = user_cont['avg_rating']
    user_profile['activity'] = user_cont['activity']

    item_cont = encoder.transform_continuous(item_profile, ['release_year_orig', 'avg_rating', 'revenue', 'budget', 'vote_count_ml'])
    item_profile['release_year_val'] = item_cont['release_year_orig'] # Normalized year
    item_profile['avg_rating'] = item_cont['avg_rating']
    item_profile['revenue'] = item_cont['revenue']
    item_profile['budget'] = item_cont['budget']
    item_profile['vote_count_ml'] = item_cont['vote_count_ml']
    
    return user_profile, item_profile

def train_dual_tower():
    print("Loading data...")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    
    with open(Path(FEATURE_STORE_DIR) / "popularity_list.json", "r") as f:
        popularity_list = json.load(f)

    # 1. Load Encoder (from Step 2)
    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()

    # 2. Apply Encoding
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)
    
    # Filter training data: only positive samples for InfoNCE
    train_pos = train_data[train_data['rating'] >= 3.0].copy()
    
    # Map movieId/userId in train_data to encoded indices
    movie_vocab = encoder.vocabularies['movieId']
    user_vocab = encoder.vocabularies['userId']
    train_pos['movieId'] = train_pos['movieId'].map(movie_vocab).fillna(0).astype(int)
    train_pos['userId'] = train_pos['userId'].map(user_vocab).fillna(0).astype(int)

    print("Calculating Log-Q for Correction...")
    item_counts = train_pos['movieId'].value_counts()
    total_count = len(train_pos)
    max_id = encoder.vocab_sizes['movieId']
    log_q_array = np.full(max_id + 1, np.log(1e-10), dtype=np.float32)
    
    encoded_ids = item_counts.index.values
    counts = item_counts.values
    log_q_array[encoded_ids] = np.log((counts / total_count) + 1e-10)
    item_profile['log_q'] = log_q_array[item_profile['movieId'].values]

    # Pre-encode Top Items for Hard Negatives
    encoded_pop_list = [movie_vocab[mid] for mid in popularity_list[:1000] if mid in movie_vocab]

    print("Preparing Global Item Features Lookup...")
    item_profile_sorted = item_profile.sort_values('movieId')
    item_lookup = {
        'item_id': torch.tensor(item_profile_sorted['movieId'].values, dtype=torch.long),
        'release_year': torch.tensor(item_profile_sorted['release_year_val'].values, dtype=torch.float32),
        'avg_rating': torch.tensor(item_profile_sorted['avg_rating'].values, dtype=torch.float32),
        'revenue': torch.tensor(item_profile_sorted['revenue'].values, dtype=torch.float32),
        'tmdb_genres': torch.tensor(np.stack(item_profile_sorted['tmdb_genres'].values), dtype=torch.long),
        'log_q': torch.tensor(item_profile_sorted['log_q'].values, dtype=torch.float32)
    }

    print("Building DataLoader...")
    dataloader = create_dataloader(train_pos, user_profile, item_profile, batch_size=1024)

    print("Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    item_lookup = {k: v.to(device) for k, v in item_lookup.items()}
    model = DualTowerModel(vocab_sizes=encoder.vocab_sizes, embed_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting Training...")
    epochs = 3
    num_simple_neg = 512
    num_hard_neg = 128
    all_item_indices = np.arange(len(item_profile))

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for user_feat, item_feat in pbar:
            user_feat = {k: v.to(device) for k, v in user_feat.items()}
            item_feat = {k: v.to(device) for k, v in item_feat.items()}
            
            simple_neg_idx = np.random.choice(all_item_indices, num_simple_neg, replace=True)
            hard_neg_idx = np.random.choice(encoded_pop_list, num_hard_neg, replace=True)
            
            def get_batch_neg(indices):
                return {k: v[indices] for k, v in item_lookup.items() if k != 'log_q'}

            extra_neg_feat = get_batch_neg(simple_neg_idx)
            extra_neg_log_q = item_lookup['log_q'][simple_neg_idx]
            hard_neg_feat = get_batch_neg(hard_neg_idx)
            
            optimizer.zero_grad()
            loss, l_infonce, l_bpr = model.compute_loss(
                user_feat, item_feat, item_log_q=item_feat['log_q'], 
                simple_neg_features=extra_neg_feat, simple_neg_log_q=extra_neg_log_q,
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

# --- ItemCF & UserCF unchanged but remain in CLI ---
def train_item_cf():
    print("Loading data for ItemCF...")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    train_data = train_data[train_data['rating'] >= 3.0]
    model = ItemCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "item_sim_matrix.pkl")
    model.fit(train_df=train_data)

def train_user_cf():
    print("Loading data for UserCF...")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    train_data = train_data[train_data['rating'] >= 3.0]
    model = UserCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "user_sim_matrix.pkl")
    model.fit(train_df=train_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train recall models.")
    parser.add_argument("--model", type=str, default="dual_tower", choices=["dual_tower", "item_cf", "user_cf"])
    args = parser.parse_args()

    if args.model == "dual_tower":
        train_dual_tower()
    elif args.model == "item_cf":
        train_item_cf()
    elif args.model == "user_cf":
        train_user_cf()
