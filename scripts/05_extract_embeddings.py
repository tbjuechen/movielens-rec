"""Extract pre-trained embeddings from the Dual-Tower model for ranking input."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import (
    PROCESSED_DATA_DIR, FEATURE_STORE_DIR, MODEL_WEIGHTS_DIR,
    EMBEDDING_DIM, TAU, TIME_DECAY_LAMBDA, BPR_GAMMA, BPR_MARGIN,
    LOSS_INFONCE_WEIGHT, LOSS_BPR_WEIGHT, LOGIT_SCALE_MAX, CONT_BUCKET_SIZE,
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN,
)
from src.features.encoder import FeatureEncoder
from src.models.recall.dual_tower import DualTowerModel


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


def pad_ts_diff(ts_list, max_len):
    if not isinstance(ts_list, (list, np.ndarray)):
        return [0.0] * max_len
    ts = list(ts_list)[:max_len]
    return ts + [0.0] * (max_len - len(ts))


def main():
    print("=== Extracting Dual-Tower Embeddings ===")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")

    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    model = DualTowerModel(
        vocab_sizes=encoder.vocab_sizes, embed_dim=EMBEDDING_DIM, tau=TAU,
        time_decay_lambda=TIME_DECAY_LAMBDA, bpr_gamma=BPR_GAMMA, bpr_margin=BPR_MARGIN,
        loss_infonce_weight=LOSS_INFONCE_WEIGHT, loss_bpr_weight=LOSS_BPR_WEIGHT,
        logit_scale_max=LOGIT_SCALE_MAX, cont_bucket_size=CONT_BUCKET_SIZE
    ).to(device)

    model_path = Path(MODEL_WEIGHTS_DIR) / "dual_tower.pth"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Train the dual-tower model first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Extract item embeddings ---
    print("Computing item embeddings...")
    max_iid = int(item_profile['movieId'].max())
    item_emb_all = np.zeros((max_iid + 1, EMBEDDING_DIM), dtype=np.float32)

    item_profile_sorted = item_profile.sort_values('movieId')
    item_feat = {
        'item_id': torch.tensor(item_profile_sorted['movieId_encoded'].values, dtype=torch.long).to(device),
        'release_year': torch.tensor(item_profile_sorted['release_year_norm'].values, dtype=torch.float32).to(device),
        'avg_rating': torch.tensor(item_profile_sorted['avg_rating_norm'].values, dtype=torch.float32).to(device),
        'revenue': torch.tensor(item_profile_sorted['revenue_norm'].values, dtype=torch.float32).to(device),
        'tmdb_genres': torch.tensor(np.stack(item_profile_sorted['tmdb_genres_encoded'].values), dtype=torch.long).to(device),
    }
    with torch.no_grad():
        _, embs = model(None, item_feat)
        raw_ids = item_profile_sorted['movieId'].values
        item_emb_all[raw_ids] = embs.cpu().numpy()
    print(f"Item embeddings: shape {item_emb_all.shape}")

    # --- Extract user embeddings (batched) ---
    print("Computing user embeddings...")
    max_uid = int(user_profile['userId'].max())
    user_emb_all = np.zeros((max_uid + 1, EMBEDDING_DIM), dtype=np.float32)

    BATCH = 1024
    user_rows = user_profile.to_dict('records')
    for start in tqdm(range(0, len(user_rows), BATCH), desc="User batches"):
        batch_rows = user_rows[start:start + BATCH]
        uids = [r['userId'] for r in batch_rows]

        user_feat = {
            'user_id': torch.tensor([r['userId_encoded'] for r in batch_rows], dtype=torch.long).to(device),
            'avg_rating': torch.tensor([r['avg_rating_norm'] for r in batch_rows], dtype=torch.float32).to(device),
            'activity': torch.tensor([r['activity_norm'] for r in batch_rows], dtype=torch.float32).to(device),
            'history': torch.tensor([r['history_encoded'] for r in batch_rows], dtype=torch.long).to(device),
            'history_ts_diff': torch.tensor(
                [pad_ts_diff(r['history_ts_diff'], USER_HISTORY_MAX_LEN) for r in batch_rows],
                dtype=torch.float32
            ).to(device),
            'top_genres': torch.tensor([r['top_genres_encoded'] for r in batch_rows], dtype=torch.long).to(device),
        }
        with torch.no_grad():
            embs, _ = model(user_feat, None)
            user_emb_all[uids] = embs.cpu().numpy()

    # --- Save ---
    out_dir = Path(FEATURE_STORE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pretrained_user_emb.npy", user_emb_all)
    np.save(out_dir / "pretrained_item_emb.npy", item_emb_all)
    print(f"Saved: pretrained_user_emb.npy {user_emb_all.shape}, pretrained_item_emb.npy {item_emb_all.shape}")


if __name__ == "__main__":
    main()
