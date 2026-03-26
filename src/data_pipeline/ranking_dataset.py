import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config.settings import (
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN
)


class RankingDataset(Dataset):
    """Pre-materialized ranking Dataset for maximum GPU utilization.

    All features are pre-expanded to sample-level contiguous tensors in __init__,
    so collate_fn is just a single contiguous slice — no indirect lookups at all.
    Trades ~2GB RAM for eliminating CPU bottleneck during training.
    """

    def __init__(self, samples, user_profile_df, item_profile_df):
        n = len(samples)
        self.n = n
        print(f"  Pre-materializing {n:,} samples into contiguous tensors...")

        uids = samples[:, 0].astype(np.int64)
        iids = samples[:, 1].astype(np.int64)

        # --- Build lookup tables (temporary, used only for pre-expansion) ---
        max_u = int(user_profile_df['userId'].max())
        max_i = int(item_profile_df['movieId'].max())

        u_idx = user_profile_df['userId'].values
        user_avg_rating = np.zeros(max_u + 1, dtype=np.float32)
        user_activity = np.zeros(max_u + 1, dtype=np.float32)
        user_top_genres = np.zeros((max_u + 1, USER_TOP_GENRES_MAX_LEN), dtype=np.int64)
        user_encoded_id = np.zeros(max_u + 1, dtype=np.int64)
        user_avg_rating[u_idx] = user_profile_df['avg_rating_norm'].values
        user_activity[u_idx] = user_profile_df['activity_norm'].values
        user_top_genres[u_idx] = np.stack(user_profile_df['top_genres_encoded'].values)
        user_encoded_id[u_idx] = user_profile_df['userId_encoded'].values

        i_idx = item_profile_df['movieId'].values
        item_release_year = np.zeros(max_i + 1, dtype=np.float32)
        item_avg_rating = np.zeros(max_i + 1, dtype=np.float32)
        item_revenue = np.zeros(max_i + 1, dtype=np.float32)
        item_budget = np.zeros(max_i + 1, dtype=np.float32)
        item_vote_count = np.zeros(max_i + 1, dtype=np.float32)
        item_genres = np.zeros((max_i + 1, ITEM_GENRES_MAX_LEN), dtype=np.int64)
        item_encoded_id = np.zeros(max_i + 1, dtype=np.int64)
        item_release_year[i_idx] = item_profile_df['release_year_norm'].values
        item_avg_rating[i_idx] = item_profile_df['avg_rating_norm'].values
        item_revenue[i_idx] = item_profile_df['revenue_norm'].values
        item_budget[i_idx] = item_profile_df['budget_norm'].values
        item_vote_count[i_idx] = item_profile_df['vote_count_ml_norm'].values
        item_genres[i_idx] = np.stack(item_profile_df['tmdb_genres_encoded'].values)
        item_encoded_id[i_idx] = item_profile_df['movieId_encoded'].values

        # --- Pre-expand all features to sample-level (N,) or (N, L) tensors ---
        # After this, lookup tables are GC'd — only flat tensors remain.
        self.user_id = torch.from_numpy(user_encoded_id[uids])
        self.item_id = torch.from_numpy(item_encoded_id[iids])
        self.user_top_genres = torch.from_numpy(user_top_genres[uids])
        self.item_genres = torch.from_numpy(item_genres[iids])
        self.user_avg_rating = torch.from_numpy(user_avg_rating[uids])
        self.user_activity = torch.from_numpy(user_activity[uids])
        self.item_release_year = torch.from_numpy(item_release_year[iids])
        self.item_avg_rating = torch.from_numpy(item_avg_rating[iids])
        self.item_revenue = torch.from_numpy(item_revenue[iids])
        self.item_budget = torch.from_numpy(item_budget[iids])
        self.item_vote_count = torch.from_numpy(item_vote_count[iids])
        self.ctr_label = torch.from_numpy(samples[:, 2].astype(np.float32))
        self.rating_label = torch.from_numpy(samples[:, 3].astype(np.float32))
        self.has_rating = torch.from_numpy((samples[:, 4] > 0.5).astype(np.bool_))

        print(f"  Pre-materialization done.")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, indices):
        """Batch collate: contiguous slice on pre-materialized tensors."""
        idx = torch.tensor(indices, dtype=torch.long)
        return {
            'user_id': self.user_id[idx],
            'item_id': self.item_id[idx],
            'user_top_genres': self.user_top_genres[idx],
            'item_genres': self.item_genres[idx],
            'user_avg_rating': self.user_avg_rating[idx],
            'user_activity': self.user_activity[idx],
            'item_release_year': self.item_release_year[idx],
            'item_avg_rating': self.item_avg_rating[idx],
            'item_revenue': self.item_revenue[idx],
            'item_budget': self.item_budget[idx],
            'item_vote_count': self.item_vote_count[idx],
            'ctr_label': self.ctr_label[idx],
            'rating_label': self.rating_label[idx],
            'has_rating': self.has_rating[idx],
        }


def build_ranking_samples(train_data, all_item_ids, neg_sample_ratio=3, seed=42):
    """Construct training samples for ranking model.

    Returns ndarray of shape (N, 5): [userId, movieId, ctr_label, rating_norm, has_rating]
    """
    rng = np.random.RandomState(seed)

    # Positive samples: rating >= 3.0
    pos_mask = train_data['rating'] >= 3.0
    pos = train_data[pos_mask][['userId', 'movieId', 'rating']].copy()
    pos['ctr_label'] = 1.0
    pos['rating_norm'] = pos['rating'] / 5.0
    pos['has_rating'] = 1.0

    # Explicit negative samples: rating < 3.0
    neg_explicit = train_data[~pos_mask][['userId', 'movieId', 'rating']].copy()
    neg_explicit['ctr_label'] = 0.0
    neg_explicit['rating_norm'] = neg_explicit['rating'] / 5.0
    neg_explicit['has_rating'] = 1.0

    # Implicit negative samples: vectorized random sampling
    # Collision rate ≈ avg_interactions/n_items ≈ 160/87K ≈ 0.2%, negligible for training
    all_items_arr = np.array(all_item_ids)
    n_implicit = int(len(pos) * neg_sample_ratio)
    print(f"Sampling {n_implicit:,} implicit negatives (vectorized)...")

    implicit_users = rng.choice(pos['userId'].values, size=n_implicit, replace=True)
    implicit_items = rng.choice(all_items_arr, size=n_implicit)
    print(f"Implicit negative sampling done.")

    neg_implicit = np.column_stack([
        implicit_users,
        implicit_items,
        np.zeros(n_implicit),      # ctr_label
        np.zeros(n_implicit),      # rating_norm (placeholder)
        np.zeros(n_implicit),      # has_rating = False
    ])

    # Combine all samples
    cols = ['userId', 'movieId', 'ctr_label', 'rating_norm', 'has_rating']
    explicit = np.vstack([
        pos[cols].values,
        neg_explicit[cols].values,
    ])
    all_samples = np.vstack([explicit, neg_implicit])

    # Shuffle
    perm = rng.permutation(len(all_samples))
    return all_samples[perm]
