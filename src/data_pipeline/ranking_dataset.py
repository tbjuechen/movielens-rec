import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config.settings import (
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN
)


class RankingDataset(Dataset):
    """Lazy-lookup ranking Dataset with DYNAMIC online negative sampling.

    Stores the positive interactions and a pool of 100 recall candidates for each.
    In each batch, it randomly samples 3 negatives from the pool for every positive.
    """

    def __init__(self, samples, user_profile_df, item_profile_df, neg_size=3):
        self.neg_size = neg_size
        
        if isinstance(samples, dict):
            # Input is from ranking_candidate_pool.parquet
            n = len(samples['userId'])
            self.n = n
            print(f"  Loading candidate pool for {n:,} positives...")
            self.uids_pos = samples['userId'].share_memory_()
            self.iids_pos = samples['movieId'].share_memory_()
            self.ctr_labels_pos = samples['ctr_label'].share_memory_()
            self.rating_norms_pos = samples['rating_norm'].share_memory_()
            # The pool: (N, 100) matrix
            self.candidate_pool = samples['candidate_pool'].share_memory_()
        else:
            # Fallback for old flat format
            raise ValueError("RankingDataset now requires the candidate pool format.")

        # --- Build lookup tables (Shared Memory) ---
        max_u = int(user_profile_df['userId'].max())
        max_i = int(item_profile_df['movieId'].max())
        u_idx = user_profile_df['userId'].values
        i_idx = item_profile_df['movieId'].values

        print(f"  Processing lookup tables (Shared Memory)...")
        # User tables
        user_encoded_id = np.zeros(max_u + 1, dtype=np.int64)
        user_avg_rating = np.zeros(max_u + 1, dtype=np.float32)
        user_activity = np.zeros(max_u + 1, dtype=np.float32)
        user_encoded_id[u_idx] = user_profile_df['userId_encoded'].values
        user_avg_rating[u_idx] = user_profile_df['avg_rating_norm'].values
        user_activity[u_idx] = user_profile_df['activity_norm'].values
        
        user_genres_list = user_profile_df['top_genres_encoded'].tolist()
        user_top_genres = np.zeros((max_u + 1, USER_TOP_GENRES_MAX_LEN), dtype=np.int64)
        user_top_genres[u_idx] = np.array(user_genres_list, dtype=np.int64)
        
        # User shared tensors
        self.user_encoded_id = torch.from_numpy(user_encoded_id).share_memory_()
        self.user_top_genres = torch.from_numpy(user_top_genres).share_memory_()
        self.user_cont_table = torch.from_numpy(np.stack([user_avg_rating, user_activity], axis=1)).share_memory_()

        # Item tables
        item_encoded_id = np.zeros(max_i + 1, dtype=np.int64)
        item_release_year = np.zeros(max_i + 1, dtype=np.float32)
        item_avg_rating = np.zeros(max_i + 1, dtype=np.float32)
        item_revenue = np.zeros(max_i + 1, dtype=np.float32)
        item_budget = np.zeros(max_i + 1, dtype=np.float32)
        item_vote_count = np.zeros(max_i + 1, dtype=np.float32)
        
        item_encoded_id[i_idx] = item_profile_df['movieId_encoded'].values
        item_release_year[i_idx] = item_profile_df['release_year_norm'].values
        item_avg_rating[i_idx] = item_profile_df['avg_rating_norm'].values
        item_revenue[i_idx] = item_profile_df['revenue_norm'].values
        item_budget[i_idx] = item_profile_df['budget_norm'].values
        item_vote_count[i_idx] = item_profile_df['vote_count_ml_norm'].values
        
        item_genres_list = item_profile_df['tmdb_genres_encoded'].tolist()
        item_genres = np.zeros((max_i + 1, ITEM_GENRES_MAX_LEN), dtype=np.int64)
        item_genres[i_idx] = np.array(item_genres_list, dtype=np.int64)

        # Item shared tensors
        self.item_encoded_id = torch.from_numpy(item_encoded_id).share_memory_()
        self.item_genres = torch.from_numpy(item_genres).share_memory_()
        self.item_cont_table = torch.from_numpy(np.stack([
            item_release_year, item_avg_rating, item_revenue, item_budget, item_vote_count
        ], axis=1)).share_memory_()

        print(f"  Lookup tables built and shared. Pool memory: ~{self.candidate_pool.nbytes / 1e9:.1f}GB")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, indices):
        """Dynamic sampling: 1 positive + neg_size negatives per interaction."""
        batch_idx = torch.as_tensor(indices, dtype=torch.long)
        batch_size = len(batch_idx)
        
        # 1. Positive IDs
        uids_pos = self.uids_pos[batch_idx]
        iids_pos = self.iids_pos[batch_idx]
        
        # 2. Randomly sample negatives from the pool (Dynamic!)
        pool_size = self.candidate_pool.shape[1]
        # Generate random indices for sampling (batch_size, neg_size)
        rel_neg_idx = torch.randint(0, pool_size, (batch_size, self.neg_size))
        # Gather movieId from pool
        iids_neg = torch.gather(self.candidate_pool[batch_idx], 1, rel_neg_idx)
        
        # 3. Concatenate (All Positives then All Negatives)
        # Final batch shape: (batch_size * (1 + neg_size), ...)
        uids = torch.cat([uids_pos, uids_pos.repeat_interleave(self.neg_size)])
        iids = torch.cat([iids_pos, iids_neg.view(-1)])
        
        # 4. Construct Labels
        ctr_label = torch.cat([
            self.ctr_labels_pos[batch_idx],
            torch.zeros(batch_size * self.neg_size)
        ])
        rating_label = torch.cat([
            self.rating_norms_pos[batch_idx],
            torch.zeros(batch_size * self.neg_size)
        ])
        has_rating = torch.cat([
            torch.ones(batch_size, dtype=torch.bool),
            torch.zeros(batch_size * self.neg_size, dtype=torch.bool)
        ])

        # 5. Fast Lookup and MERGE everything into 3 blocks to minimize IPC
        # Block 1: Int features (user_id, item_id, user_genres, item_genres) -> (Total_B, 1 + 1 + 10 + 10 = 22)
        int_features = torch.cat([
            self.user_encoded_id[uids].unsqueeze(1),
            self.item_encoded_id[iids].unsqueeze(1),
            self.user_top_genres[uids],
            self.item_genres[iids]
        ], dim=1)

        # Block 2: Float features (cont_features) -> (Total_B, 7)
        float_features = torch.cat([
            self.user_cont_table[uids],
            self.item_cont_table[iids]
        ], dim=1)

        # Block 3: Labels and Mask -> (Total_B, 3)
        # 0: ctr, 1: rating, 2: has_rating_mask
        labels = torch.stack([
            ctr_label,
            rating_label,
            has_rating.float()
        ], dim=1)

        return {
            'int_features': int_features,
            'float_features': float_features,
            'labels': labels
        }


def build_ranking_samples(train_data, all_item_ids, neg_sample_ratio=3, seed=42):
    """Construct training samples for ranking model.

    Returns dict of ndarrays: [userId, movieId, ctr_label, rating_norm, has_rating]
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

    # Combine all samples while keeping them separate
    uids = np.concatenate([pos['userId'].values, neg_explicit['userId'].values, implicit_users])
    iids = np.concatenate([pos['movieId'].values, neg_explicit['movieId'].values, implicit_items])
    ctr_labels = np.concatenate([pos['ctr_label'].values, neg_explicit['ctr_label'].values, np.zeros(n_implicit)])
    rating_norms = np.concatenate([pos['rating_norm'].values, neg_explicit['rating_norm'].values, np.zeros(n_implicit)])
    has_ratings = np.concatenate([pos['has_rating'].values, neg_explicit['has_rating'].values, np.zeros(n_implicit)])

    # Shuffle everything together
    perm = rng.permutation(len(uids))
    return {
        'userId': uids[perm],
        'movieId': iids[perm],
        'ctr_label': ctr_labels[perm],
        'rating_norm': rating_norms[perm],
        'has_rating': has_ratings[perm],
    }
