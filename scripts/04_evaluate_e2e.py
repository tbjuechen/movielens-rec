import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import (
    PROCESSED_DATA_DIR, FEATURE_STORE_DIR, MODEL_WEIGHTS_DIR,
    EMBEDDING_DIM, RECALL_K
)
from src.features.encoder import FeatureEncoder
from src.models.recall.dual_tower import DualTowerModel
from src.models.recall.item_cf import ItemCFModel
from src.models.recall.user_cf import UserCFModel
from src.models.recall.simple_recall import PopularityRecall, GenreRecall
from src.models.recall.merger import RecallMerger
from src.evaluation.metrics import recall_at_k, ndcg_at_k

def evaluate():
    print("=== Starting End-to-End Evaluation ===")
    
    # 1. Load Data
    print("Loading data and encoders...")
    val_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "val_data.parquet")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    
    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()
    
    # Reuse the encoding logic from training
    from scripts.03_train_models import apply_encoding
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 2. Initialize Models
    print("Initializing models...")
    # Dual Tower
    model = DualTowerModel(vocab_sizes=encoder.vocab_sizes, embed_dim=EMBEDDING_DIM).to(device)
    model_path = Path(MODEL_WEIGHTS_DIR) / "dual_tower.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_to=device))
        print("Loaded Dual-Tower weights.")
    model.eval()

    # ItemCF
    item_cf = ItemCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "item_sim_matrix.pkl")
    try:
        item_cf.load()
    except: print("Warning: ItemCF matrix not found.")

    # UserCF
    user_cf = UserCFModel(sim_save_path=Path(MODEL_WEIGHTS_DIR) / "user_sim_matrix.pkl")
    try:
        user_cf.load()
    except: print("Warning: UserCF matrix not found.")

    # Simple Recalls
    pop_recall = PopularityRecall(item_profile_path=Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    genre_recall = GenreRecall(
        genre_to_items_path=Path(FEATURE_STORE_DIR) / "genre_to_items.json",
        item_profile_path=Path(PROCESSED_DATA_DIR) / "item_profile.parquet"
    )
    
    merger = RecallMerger(top_k=RECALL_K)

    # 3. Export Item Embeddings and Build FAISS Index
    print("Generating Item Embeddings for full library...")
    # Prepare full item tensors
    item_profile = item_profile.sort_values('movieId_encoded')
    item_feat_tensors = {
        'item_id': torch.tensor(item_profile['movieId_encoded'].values, dtype=torch.long).to(device),
        'release_year': torch.tensor(item_profile['release_year_norm'].values, dtype=torch.float32).to(device),
        'avg_rating': torch.tensor(item_profile['avg_rating_norm'].values, dtype=torch.float32).to(device),
        'revenue': torch.tensor(item_profile['revenue_norm'].values, dtype=torch.float32).to(device),
        'tmdb_genres': torch.tensor(np.stack(item_profile['tmdb_genres_encoded'].values), dtype=torch.long).to(device)
    }
    
    with torch.no_grad():
        _, all_item_embs = model(None, item_feat_tensors)
        all_item_embs = all_item_embs.cpu().numpy().astype('float32')

    # Build Index (Inner Product)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(all_item_embs)
    print(f"FAISS index built with {index.ntotal} items.")

    # Mapping back to original movieId
    idx_to_movie_id = item_profile['movieId'].values

    # 4. Perform Multi-channel Recall for Validation Users
    print("Running evaluation on validation users...")
    results = []
    
    # Select a sample or full val_data (for speed, you can limit this)
    eval_users = val_data['userId'].unique()
    # We select up to 5000 users for quick validation
    eval_users = eval_users[:5000]
    
    # Prepare User Profiles for fast lookup
    user_profile = user_profile.set_index('userId')
    
    for uid in tqdm(eval_users, desc="Recall Channels"):
        user_row = user_profile.loc[uid]
        actual_item = val_data[val_data['userId'] == uid]['movieId'].tolist()
        
        user_channels = {}
        
        # --- Channel 1: Dual Tower ---
        user_tensor = {
            'user_id': torch.tensor([user_row['userId_encoded']], dtype=torch.long).to(device),
            'avg_rating': torch.tensor([user_row['avg_rating_norm']], dtype=torch.float32).to(device),
            'activity': torch.tensor([user_row['activity_norm']], dtype=torch.float32).to(device),
            'history': torch.tensor([user_row['history_encoded']], dtype=torch.long).to(device),
            'history_ts_diff': torch.tensor([user_row['history_ts_diff']], dtype=torch.float32).to(device),
            'top_genres': torch.tensor([user_row['top_genres_encoded']], dtype=torch.long).to(device)
        }
        with torch.no_grad():
            u_emb, _ = model(user_tensor, None)
            u_emb = u_emb.cpu().numpy().astype('float32')
        
        D, I = index.search(u_emb, RECALL_K)
        user_channels['dual_tower'] = [idx_to_movie_id[idx] for idx in I[0]]
        
        # --- Channel 2: ItemCF ---
        user_channels['item_cf'] = item_cf.retrieve(user_row['history_encoded'], k=RECALL_K)
        
        # --- Channel 3: Genre ---
        user_channels['genre'] = genre_recall.retrieve(user_row['top_genres'], k=RECALL_K)
        
        # --- Channel 4: Popularity ---
        user_channels['popularity'] = pop_recall.retrieve(k=RECALL_K)
        
        # --- Merge ---
        merged_res = merger.merge(user_channels)
        
        results.append({
            'actual': actual_item,
            'predicted': merged_res,
            'predicted_dual': user_channels['dual_tower']
        })

    # 5. Compute Metrics
    print("\n=== Metrics Report ===")
    metrics = {
        'Recall@50': [], 'NDCG@50': [],
        'Recall@50 (Dual-Tower Only)': []
    }
    
    for res in results:
        metrics['Recall@50'].append(recall_at_k(res['actual'], res['predicted'], k=50))
        metrics['NDCG@50'].append(ndcg_at_k(res['actual'], res['predicted'], k=50))
        metrics['Recall@50 (Dual-Tower Only)'].append(recall_at_k(res['actual'], res['predicted_dual'], k=50))
        
    for m, vals in metrics.items():
        print(f"{m}: {np.mean(vals):.4f}")

if __name__ == "__main__":
    evaluate()
