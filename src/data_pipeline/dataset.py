import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from src.config.settings import (
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN, BATCH_SIZE
)

class MovielensRecallDataset(Dataset):
    def __init__(self, interactions_df: pd.DataFrame, user_profile_df: pd.DataFrame, item_profile_df: pd.DataFrame):
        # 1. 预处理 Interactions 为 Numpy 提高读取速度
        self.u_ids = interactions_df['userId'].values.astype(np.int32)
        self.i_ids = interactions_df['movieId'].values.astype(np.int32)
        
        # 2. 预处理 Profiles 为快速索引数组
        # 使用原始 ID 索引特征数组, 同时维护 encoded ID 映射供 Embedding 层使用
        max_u = user_profile_df['userId'].max()
        max_i = item_profile_df['movieId'].max()

        # 创建固定长度的矩阵/数组存储特征，实现 O(1) 访问
        self.user_avg_rating = np.zeros(max_u + 1, dtype=np.float32)
        self.user_activity = np.zeros(max_u + 1, dtype=np.float32)
        self.user_history = np.zeros((max_u + 1, USER_HISTORY_MAX_LEN), dtype=np.int32)
        self.user_history_ts = np.zeros((max_u + 1, USER_HISTORY_MAX_LEN), dtype=np.float32)
        self.user_top_genres = np.zeros((max_u + 1, USER_TOP_GENRES_MAX_LEN), dtype=np.int32)

        # 填充数据
        u_idx = user_profile_df['userId'].values
        self.user_avg_rating[u_idx] = user_profile_df['avg_rating'].values
        self.user_activity[u_idx] = user_profile_df['activity'].values
        self.user_history[u_idx] = np.stack(user_profile_df['history'].values)
        # Pad history_ts_diff to fixed length (not encoded by categorical encoder)
        ts_lists = user_profile_df['history_ts_diff'].values
        padded_ts = np.zeros((len(ts_lists), USER_HISTORY_MAX_LEN), dtype=np.float32)
        for i, ts in enumerate(ts_lists):
            length = min(len(ts), USER_HISTORY_MAX_LEN)
            padded_ts[i, :length] = ts[:length]
        self.user_history_ts[u_idx] = padded_ts
        self.user_top_genres[u_idx] = np.stack(user_profile_df['top_genres'].values)

        # Encoded ID mapping (for model embedding lookup)
        self.user_encoded_id = np.zeros(max_u + 1, dtype=np.int32)
        self.user_encoded_id[u_idx] = user_profile_df['user_id'].values
        
        # Item Side
        self.item_release_year = np.zeros(max_i + 1, dtype=np.float32)
        self.item_avg_rating = np.zeros(max_i + 1, dtype=np.float32)
        self.item_revenue = np.zeros(max_i + 1, dtype=np.float32)
        self.item_genres = np.zeros((max_i + 1, ITEM_GENRES_MAX_LEN), dtype=np.int32)
        self.item_log_q = np.full(max_i + 1, np.log(1e-10), dtype=np.float32)
        
        i_idx = item_profile_df['movieId'].values
        self.item_release_year[i_idx] = item_profile_df['release_year_val'].values
        self.item_avg_rating[i_idx] = item_profile_df['avg_rating'].values
        self.item_revenue[i_idx] = item_profile_df['revenue'].values
        self.item_genres[i_idx] = np.stack(item_profile_df['tmdb_genres'].values)
        self.item_log_q[i_idx] = item_profile_df['log_q'].values

        # Encoded ID mapping (for model embedding lookup)
        self.item_encoded_id = np.zeros(max_i + 1, dtype=np.int32)
        self.item_encoded_id[i_idx] = item_profile_df['item_id'].values

    def __len__(self):
        return len(self.u_ids)

    def __getitem__(self, idx):
        uid = self.u_ids[idx]
        iid = self.i_ids[idx]
        
        user_tensor_dict = {
            'user_id': torch.tensor(self.user_encoded_id[uid], dtype=torch.long),
            'avg_rating': torch.tensor(self.user_avg_rating[uid], dtype=torch.float32),
            'activity': torch.tensor(self.user_activity[uid], dtype=torch.float32),
            'history': torch.tensor(self.user_history[uid], dtype=torch.long),
            'history_ts_diff': torch.tensor(self.user_history_ts[uid], dtype=torch.float32),
            'top_genres': torch.tensor(self.user_top_genres[uid], dtype=torch.long)
        }
        
        item_tensor_dict = {
            'item_id': torch.tensor(self.item_encoded_id[iid], dtype=torch.long),
            'release_year': torch.tensor(self.item_release_year[iid], dtype=torch.float32),
            'avg_rating': torch.tensor(self.item_avg_rating[iid], dtype=torch.float32),
            'revenue': torch.tensor(self.item_revenue[iid], dtype=torch.float32),
            'tmdb_genres': torch.tensor(self.item_genres[iid], dtype=torch.long),
            'log_q': torch.tensor(self.item_log_q[iid], dtype=torch.float32)
        }
        
        return user_tensor_dict, item_tensor_dict

def create_dataloader(interactions, user_profile, item_profile, batch_size=BATCH_SIZE, shuffle=True, num_workers=4):
    dataset = MovielensRecallDataset(interactions, user_profile, item_profile)
    # 增加 persistent_workers 防止子进程内存反复初始化开销
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=(num_workers > 0))
