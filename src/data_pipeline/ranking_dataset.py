import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config.settings import (
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN
)


class RankingDataset(Dataset):
    """Lazy-lookup ranking Dataset for large-scale training.

    Stores only the raw sample array (N, 5) in memory (~40GB for 1.3B samples).
    Feature lookup is done at collate time via pre-built numpy arrays indexed by
    raw userId/movieId, avoiding the 200GB+ cost of pre-materialization.
    """

    def __init__(self, samples, user_profile_df, item_profile_df):
        n = len(samples)
        self.n = n
        print(f"  Building lookup tables for {n:,} samples...")

        # Store raw sample columns as contiguous arrays
        self.uids = samples[:, 0].astype(np.int64)
        self.iids = samples[:, 1].astype(np.int64)
        self.ctr_label = samples[:, 2].astype(np.float32)
        self.rating_label = samples[:, 3].astype(np.float32)
        self.has_rating = (samples[:, 4] > 0.5)

        # --- Build lookup tables (shared across all collate calls) ---
        max_u = int(user_profile_df['userId'].max())
        max_i = int(item_profile_df['movieId'].max())

        u_idx = user_profile_df['userId'].values
        self.user_encoded_id = np.zeros(max_u + 1, dtype=np.int64)
        self.user_avg_rating = np.zeros(max_u + 1, dtype=np.float32)
        self.user_activity = np.zeros(max_u + 1, dtype=np.float32)
        self.user_top_genres = np.zeros((max_u + 1, USER_TOP_GENRES_MAX_LEN), dtype=np.int64)
        self.user_encoded_id[u_idx] = user_profile_df['userId_encoded'].values
        self.user_avg_rating[u_idx] = user_profile_df['avg_rating_norm'].values
        self.user_activity[u_idx] = user_profile_df['activity_norm'].values
        self.user_top_genres[u_idx] = np.stack(user_profile_df['top_genres_encoded'].values)

        i_idx = item_profile_df['movieId'].values
        self.item_encoded_id = np.zeros(max_i + 1, dtype=np.int64)
        self.item_release_year = np.zeros(max_i + 1, dtype=np.float32)
        self.item_avg_rating = np.zeros(max_i + 1, dtype=np.float32)
        self.item_revenue = np.zeros(max_i + 1, dtype=np.float32)
        self.item_budget = np.zeros(max_i + 1, dtype=np.float32)
        self.item_vote_count = np.zeros(max_i + 1, dtype=np.float32)
        self.item_genres = np.zeros((max_i + 1, ITEM_GENRES_MAX_LEN), dtype=np.int64)
        self.item_encoded_id[i_idx] = item_profile_df['movieId_encoded'].values
        self.item_release_year[i_idx] = item_profile_df['release_year_norm'].values
        self.item_avg_rating[i_idx] = item_profile_df['avg_rating_norm'].values
        self.item_revenue[i_idx] = item_profile_df['revenue_norm'].values
        self.item_budget[i_idx] = item_profile_df['budget_norm'].values
        self.item_vote_count[i_idx] = item_profile_df['vote_count_ml_norm'].values
        self.item_genres[i_idx] = np.stack(item_profile_df['tmdb_genres_encoded'].values)

        print(f"  Lookup tables built. Sample memory: ~{n * 5 * 4 / 1e9:.1f}GB")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, indices):
        """Batch collate: lookup features from pre-built numpy arrays."""
        idx = np.array(indices, dtype=np.int64)
        uids = self.uids[idx]
        iids = self.iids[idx]

        return {
            'user_id': torch.from_numpy(self.user_encoded_id[uids]),
            'item_id': torch.from_numpy(self.item_encoded_id[iids]),
            'user_top_genres': torch.from_numpy(self.user_top_genres[uids]),
            'item_genres': torch.from_numpy(self.item_genres[iids]),
            'user_avg_rating': torch.from_numpy(self.user_avg_rating[uids].copy()),
            'user_activity': torch.from_numpy(self.user_activity[uids].copy()),
            'item_release_year': torch.from_numpy(self.item_release_year[iids].copy()),
            'item_avg_rating': torch.from_numpy(self.item_avg_rating[iids].copy()),
            'item_revenue': torch.from_numpy(self.item_revenue[iids].copy()),
            'item_budget': torch.from_numpy(self.item_budget[iids].copy()),
            'item_vote_count': torch.from_numpy(self.item_vote_count[iids].copy()),
            'ctr_label': torch.from_numpy(self.ctr_label[idx].copy()),
            'rating_label': torch.from_numpy(self.rating_label[idx].copy()),
            'has_rating': torch.from_numpy(self.has_rating[idx].copy()),
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
