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
        if isinstance(samples, dict):
            # Input is already optimized dictionary of tensors
            n = len(samples['userId'])
            self.n = n
            print(f"  Using pre-loaded tensor samples: {n:,}")
            # Explicitly share memory for all sample tensors
            self.uids = samples['userId'].share_memory_()
            self.iids = samples['movieId'].share_memory_()
            self.ctr_label = samples['ctr_label'].share_memory_()
            self.rating_label = samples['rating_norm'].share_memory_()
            self.has_rating = samples['has_rating'].share_memory_()
        else:
            n = len(samples)
            self.n = n
            print(f"  Building lookup tables for {n:,} samples...")

            # Store raw sample columns as contiguous tensors and share memory
            self.uids = torch.from_numpy(samples[:, 0].astype(np.int64)).share_memory_()
            self.iids = torch.from_numpy(samples[:, 1].astype(np.int64)).share_memory_()
            self.ctr_label = torch.from_numpy(samples[:, 2].astype(np.float32)).share_memory_()
            self.rating_label = torch.from_numpy(samples[:, 3].astype(np.float32)).share_memory_()
            self.has_rating = (torch.from_numpy(samples[:, 4].astype(np.float32)) > 0.5).share_memory_()

        # --- Build lookup tables (shared across all collate calls) ---
        max_u = int(user_profile_df['userId'].max())
        max_i = int(item_profile_df['movieId'].max())

        u_idx = user_profile_df['userId'].values
        user_encoded_id = np.zeros(max_u + 1, dtype=np.int64)
        user_avg_rating = np.zeros(max_u + 1, dtype=np.float32)
        user_activity = np.zeros(max_u + 1, dtype=np.float32)
        
        print(f"  Processing user profile lookup tables...")
        user_encoded_id[u_idx] = user_profile_df['userId_encoded'].values
        user_avg_rating[u_idx] = user_profile_df['avg_rating_norm'].values
        user_activity[u_idx] = user_profile_df['activity_norm'].values
        
        user_genres_list = user_profile_df['top_genres_encoded'].tolist()
        user_top_genres = np.zeros((max_u + 1, USER_TOP_GENRES_MAX_LEN), dtype=np.int64)
        user_top_genres[u_idx] = np.array(user_genres_list, dtype=np.int64)
        del user_genres_list

        # PRE-MERGE User Continuous Features (B, 2)
        user_cont = np.stack([user_avg_rating, user_activity], axis=1)
        self.user_cont_table = torch.from_numpy(user_cont).share_memory_()

        # Convert to PyTorch Tensors and ENABLE SHARED MEMORY
        self.user_encoded_id = torch.from_numpy(user_encoded_id).share_memory_()
        self.user_top_genres = torch.from_numpy(user_top_genres).share_memory_()

        print(f"  Processing item profile lookup tables...")
        i_idx = item_profile_df['movieId'].values
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
        del item_genres_list

        # PRE-MERGE Item Continuous Features (B, 5)
        item_cont = np.stack([
            item_release_year, item_avg_rating, item_revenue, 
            item_budget, item_vote_count
        ], axis=1)
        self.item_cont_table = torch.from_numpy(item_cont).share_memory_()

        # Convert to PyTorch Tensors and ENABLE SHARED MEMORY
        self.item_encoded_id = torch.from_numpy(item_encoded_id).share_memory_()
        self.item_genres = torch.from_numpy(item_genres).share_memory_()

        print(f"  Lookup tables built and shared. Sample memory: ~{n * 5 * 4 / 1e9:.1f}GB")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, indices):
        """Batch collate: lookup and MERGE features to reduce IPC overhead."""
        idx = torch.as_tensor(indices, dtype=torch.long)
        uids = self.uids[idx]
        iids = self.iids[idx]

        # Combine all continuous features into a single block (N, 7)
        # Faster lookup from 2 pre-merged tables instead of 7
        cont_features = torch.cat([
            self.user_cont_table[uids],
            self.item_cont_table[iids]
        ], dim=1)

        return {
            'user_id': self.user_encoded_id[uids],
            'item_id': self.item_encoded_id[iids],
            'user_top_genres': self.user_top_genres[uids],
            'item_genres': self.item_genres[iids],
            'cont_features': cont_features, # Merged 7 columns into 1 tensor
            'ctr_label': self.ctr_label[idx],
            'rating_label': self.rating_label[idx],
            'has_rating': self.has_rating[idx],
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
