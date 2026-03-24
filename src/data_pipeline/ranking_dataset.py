import numpy as np
import torch
from torch.utils.data import Dataset

from src.config.settings import (
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN
)


class RankingDataset(Dataset):
    """Zero-copy PyTorch Dataset for ranking model training.

    Samples are (user_id, item_id, ctr_label, rating_label, has_rating) tuples.
    Features are looked up via pre-allocated numpy arrays indexed by raw IDs.
    """

    def __init__(self, samples, user_profile_df, item_profile_df,
                 pretrained_user_emb, pretrained_item_emb):
        """
        Args:
            samples: ndarray of shape (N, 5) — [userId, movieId, ctr_label, rating_norm, has_rating]
            user_profile_df: DataFrame with encoded user features
            item_profile_df: DataFrame with encoded item features
            pretrained_user_emb: ndarray (max_uid+1, emb_dim)
            pretrained_item_emb: ndarray (max_iid+1, emb_dim)
        """
        self.samples = samples.astype(np.float32)
        self.n = len(samples)

        max_u = int(user_profile_df['userId'].max())
        max_i = int(item_profile_df['movieId'].max())

        # --- User side ---
        u_idx = user_profile_df['userId'].values

        user_avg_rating = np.zeros(max_u + 1, dtype=np.float32)
        user_activity = np.zeros(max_u + 1, dtype=np.float32)
        user_top_genres = np.zeros((max_u + 1, USER_TOP_GENRES_MAX_LEN), dtype=np.int64)
        user_encoded_id = np.zeros(max_u + 1, dtype=np.int64)

        user_avg_rating[u_idx] = user_profile_df['avg_rating_norm'].values
        user_activity[u_idx] = user_profile_df['activity_norm'].values
        user_top_genres[u_idx] = np.stack(user_profile_df['top_genres_encoded'].values)
        user_encoded_id[u_idx] = user_profile_df['userId_encoded'].values

        self.user_avg_rating = torch.from_numpy(user_avg_rating)
        self.user_activity = torch.from_numpy(user_activity)
        self.user_top_genres = torch.from_numpy(user_top_genres)
        self.user_encoded_id = torch.from_numpy(user_encoded_id)

        # --- Item side ---
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

        self.item_release_year = torch.from_numpy(item_release_year)
        self.item_avg_rating = torch.from_numpy(item_avg_rating)
        self.item_revenue = torch.from_numpy(item_revenue)
        self.item_budget = torch.from_numpy(item_budget)
        self.item_vote_count = torch.from_numpy(item_vote_count)
        self.item_genres = torch.from_numpy(item_genres)
        self.item_encoded_id = torch.from_numpy(item_encoded_id)

        # --- Pre-trained embeddings ---
        self.pt_user_emb = torch.from_numpy(pretrained_user_emb.astype(np.float32))
        self.pt_item_emb = torch.from_numpy(pretrained_item_emb.astype(np.float32))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        uid = int(self.samples[idx, 0])
        iid = int(self.samples[idx, 1])
        ctr_label = self.samples[idx, 2]
        rating_label = self.samples[idx, 3]
        has_rating = self.samples[idx, 4]

        return {
            # Sparse
            'user_id': self.user_encoded_id[uid],
            'item_id': self.item_encoded_id[iid],
            'user_top_genres': self.user_top_genres[uid],
            'item_genres': self.item_genres[iid],
            # Continuous
            'user_avg_rating': self.user_avg_rating[uid],
            'user_activity': self.user_activity[uid],
            'item_release_year': self.item_release_year[iid],
            'item_avg_rating': self.item_avg_rating[iid],
            'item_revenue': self.item_revenue[iid],
            'item_budget': self.item_budget[iid],
            'item_vote_count': self.item_vote_count[iid],
            # Pre-trained dense
            'user_emb_pretrained': self.pt_user_emb[uid],
            'item_emb_pretrained': self.pt_item_emb[iid],
            # Labels
            'ctr_label': torch.tensor(ctr_label, dtype=torch.float32),
            'rating_label': torch.tensor(rating_label, dtype=torch.float32),
            'has_rating': torch.tensor(has_rating > 0.5, dtype=torch.bool),
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
