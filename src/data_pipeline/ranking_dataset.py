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

        # Store raw sample columns as contiguous tensors to avoid Python CoW memory duplication in DataLoader
        self.uids = torch.from_numpy(samples[:, 0].astype(np.int64))
        self.iids = torch.from_numpy(samples[:, 1].astype(np.int64))
        self.ctr_label = torch.from_numpy(samples[:, 2].astype(np.float32))
        self.rating_label = torch.from_numpy(samples[:, 3].astype(np.float32))
        self.has_rating = torch.from_numpy(samples[:, 4].astype(np.float32)) > 0.5

        # --- Build lookup tables (shared across all collate calls) ---
        max_u = int(user_profile_df['userId'].max())
        max_i = int(item_profile_df['movieId'].max())

        u_idx = user_profile_df['userId'].values
        user_encoded_id = np.zeros(max_u + 1, dtype=np.int64)
        user_avg_rating = np.zeros(max_u + 1, dtype=np.float32)
        user_activity = np.zeros(max_u + 1, dtype=np.float32)
        user_top_genres = np.zeros((max_u + 1, USER_TOP_GENRES_MAX_LEN), dtype=np.int64)
        user_encoded_id[u_idx] = user_profile_df['userId_encoded'].values
        user_avg_rating[u_idx] = user_profile_df['avg_rating_norm'].values
        user_activity[u_idx] = user_profile_df['activity_norm'].values
        user_top_genres[u_idx] = np.stack(user_profile_df['top_genres_encoded'].values)

        # Convert to PyTorch Tensors for DataLoader shared memory
        self.user_encoded_id = torch.from_numpy(user_encoded_id)
        self.user_avg_rating = torch.from_numpy(user_avg_rating)
        self.user_activity = torch.from_numpy(user_activity)
        self.user_top_genres = torch.from_numpy(user_top_genres)

        i_idx = item_profile_df['movieId'].values
        item_encoded_id = np.zeros(max_i + 1, dtype=np.int64)
        item_release_year = np.zeros(max_i + 1, dtype=np.float32)
        item_avg_rating = np.zeros(max_i + 1, dtype=np.float32)
        item_revenue = np.zeros(max_i + 1, dtype=np.float32)
        item_budget = np.zeros(max_i + 1, dtype=np.float32)
        item_vote_count = np.zeros(max_i + 1, dtype=np.float32)
        item_genres = np.zeros((max_i + 1, ITEM_GENRES_MAX_LEN), dtype=np.int64)
        item_encoded_id[i_idx] = item_profile_df['movieId_encoded'].values
        item_release_year[i_idx] = item_profile_df['release_year_norm'].values
        item_avg_rating[i_idx] = item_profile_df['avg_rating_norm'].values
        item_revenue[i_idx] = item_profile_df['revenue_norm'].values
        item_budget[i_idx] = item_profile_df['budget_norm'].values
        item_vote_count[i_idx] = item_profile_df['vote_count_ml_norm'].values
        item_genres[i_idx] = np.stack(item_profile_df['tmdb_genres_encoded'].values)

        # Convert to PyTorch Tensors for DataLoader shared memory
        self.item_encoded_id = torch.from_numpy(item_encoded_id)
        self.item_release_year = torch.from_numpy(item_release_year)
        self.item_avg_rating = torch.from_numpy(item_avg_rating)
        self.item_revenue = torch.from_numpy(item_revenue)
        self.item_budget = torch.from_numpy(item_budget)
        self.item_vote_count = torch.from_numpy(item_vote_count)
        self.item_genres = torch.from_numpy(item_genres)

        print(f"  Lookup tables built. Sample memory: ~{n * 5 * 4 / 1e9:.1f}GB")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, indices):
        """Batch collate: lookup features from pre-built tensor arrays."""
        idx = torch.tensor(indices, dtype=torch.long)
        uids = self.uids[idx]
        iids = self.iids[idx]

        return {
            'user_id': self.user_encoded_id[uids],
            'item_id': self.item_encoded_id[iids],
            'user_top_genres': self.user_top_genres[uids],
            'item_genres': self.item_genres[iids],
            'user_avg_rating': self.user_avg_rating[uids],
            'user_activity': self.user_activity[uids],
            'item_release_year': self.item_release_year[iids],
            'item_avg_rating': self.item_avg_rating[iids],
            'item_revenue': self.item_revenue[iids],
            'item_budget': self.item_budget[iids],
            'item_vote_count': self.item_vote_count[iids],
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
