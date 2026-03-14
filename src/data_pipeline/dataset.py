import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src.config.settings import (
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN,
    BATCH_SIZE, NUM_WORKERS
)

class MovielensRecallDataset(Dataset):
    def __init__(self, interactions_df, user_profile_df, item_profile_df):
        self.u_ids = interactions_df['userId'].values.astype(np.int64)
        self.i_ids = interactions_df['movieId'].values.astype(np.int64)
        self.n = len(self.u_ids)

        max_u = int(user_profile_df['userId'].max())
        max_i = int(item_profile_df['movieId'].max())

        # --- Build numpy arrays, then zero-copy convert to torch ---
        # User Side
        u_idx = user_profile_df['userId'].values

        user_avg_rating = np.zeros(max_u + 1, dtype=np.float32)
        user_activity = np.zeros(max_u + 1, dtype=np.float32)
        user_history = np.zeros((max_u + 1, USER_HISTORY_MAX_LEN), dtype=np.int64)
        user_history_ts = np.zeros((max_u + 1, USER_HISTORY_MAX_LEN), dtype=np.float32)
        user_top_genres = np.zeros((max_u + 1, USER_TOP_GENRES_MAX_LEN), dtype=np.int64)
        user_encoded_id = np.zeros(max_u + 1, dtype=np.int64)

        user_avg_rating[u_idx] = user_profile_df['avg_rating'].values
        user_activity[u_idx] = user_profile_df['activity'].values
        user_history[u_idx] = np.stack(user_profile_df['history'].values)
        ts_lists = user_profile_df['history_ts_diff'].values
        padded_ts = np.zeros((len(ts_lists), USER_HISTORY_MAX_LEN), dtype=np.float32)
        for i, ts in enumerate(ts_lists):
            length = min(len(ts), USER_HISTORY_MAX_LEN)
            padded_ts[i, :length] = ts[:length]
        user_history_ts[u_idx] = padded_ts
        user_top_genres[u_idx] = np.stack(user_profile_df['top_genres'].values)
        user_encoded_id[u_idx] = user_profile_df['user_id'].values

        # Item Side
        i_idx = item_profile_df['movieId'].values

        item_release_year = np.zeros(max_i + 1, dtype=np.float32)
        item_avg_rating = np.zeros(max_i + 1, dtype=np.float32)
        item_revenue = np.zeros(max_i + 1, dtype=np.float32)
        item_genres = np.zeros((max_i + 1, ITEM_GENRES_MAX_LEN), dtype=np.int64)
        item_log_q = np.full(max_i + 1, np.log(1e-10), dtype=np.float32)
        item_encoded_id = np.zeros(max_i + 1, dtype=np.int64)

        item_release_year[i_idx] = item_profile_df['release_year_val'].values
        item_avg_rating[i_idx] = item_profile_df['avg_rating'].values
        item_revenue[i_idx] = item_profile_df['revenue'].values
        item_genres[i_idx] = np.stack(item_profile_df['tmdb_genres'].values)
        item_log_q[i_idx] = item_profile_df['log_q'].values
        item_encoded_id[i_idx] = item_profile_df['item_id'].values

        # torch.from_numpy: zero-copy, shares memory with numpy
        # __getitem__ becomes pure tensor indexing (returns views, no allocation)
        self.user_avg_rating = torch.from_numpy(user_avg_rating)
        self.user_activity = torch.from_numpy(user_activity)
        self.user_history = torch.from_numpy(user_history)
        self.user_history_ts = torch.from_numpy(user_history_ts)
        self.user_top_genres = torch.from_numpy(user_top_genres)
        self.user_encoded_id = torch.from_numpy(user_encoded_id)

        self.item_release_year = torch.from_numpy(item_release_year)
        self.item_avg_rating = torch.from_numpy(item_avg_rating)
        self.item_revenue = torch.from_numpy(item_revenue)
        self.item_genres = torch.from_numpy(item_genres)
        self.item_log_q = torch.from_numpy(item_log_q)
        self.item_encoded_id = torch.from_numpy(item_encoded_id)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        uid = self.u_ids[idx]
        iid = self.i_ids[idx]

        return {
            'user_id': self.user_encoded_id[uid],
            'avg_rating': self.user_avg_rating[uid],
            'activity': self.user_activity[uid],
            'history': self.user_history[uid],
            'history_ts_diff': self.user_history_ts[uid],
            'top_genres': self.user_top_genres[uid]
        }, {
            'item_id': self.item_encoded_id[iid],
            'release_year': self.item_release_year[iid],
            'avg_rating': self.item_avg_rating[iid],
            'revenue': self.item_revenue[iid],
            'tmdb_genres': self.item_genres[iid],
            'log_q': self.item_log_q[iid]
        }

def create_dataloader(interactions, user_profile, item_profile, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS):
    dataset = MovielensRecallDataset(interactions, user_profile, item_profile)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
        pin_memory=True, prefetch_factor=4 if num_workers > 0 else None
    )
