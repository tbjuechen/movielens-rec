import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MovielensRecallDataset(Dataset):
    def __init__(self, interactions_df: pd.DataFrame, user_profile_df: pd.DataFrame, item_profile_df: pd.DataFrame):
        """
        初始化数据集。
        interactions_df: 已经切分好的训练集/验证集 (只包含正样本，例如 rating >= 3.0)
        """
        self.interactions = interactions_df.reset_index(drop=True)
        
        # 将 profile 转换为以 ID 为索引的字典，加快查询速度
        # 注意：这里假设 profile 已经经过了 FeatureEncoder 编码成了整数索引
        self.user_profile = user_profile_df.set_index('userId').to_dict('index')
        self.item_profile = item_profile_df.set_index('movieId').to_dict('index')

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = int(row['userId'])
        item_id = int(row['movieId'])
        
        user_feat = self.user_profile[user_id]
        item_feat = self.item_profile.get(item_id)
        
        if item_feat is None:
            # Fallback to a zero-filled dictionary if item profile is missing
            item_feat = {
                'release_year': 0.0,
                'avg_rating': 0.0,
                'revenue': 0.0,
                'tmdb_genres': [0] * 5
            }
        
        # 构建张量字典
        user_tensor_dict = {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'avg_rating': torch.tensor(user_feat['avg_rating'], dtype=torch.float32),
            'activity': torch.tensor(user_feat['activity'], dtype=torch.float32),
            'history': torch.tensor(user_feat['history'], dtype=torch.long),      # list of ints
            'top_genres': torch.tensor(user_feat['top_genres'], dtype=torch.long) # list of ints
        }
        
        item_tensor_dict = {
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'release_year': torch.tensor(item_feat['release_year'], dtype=torch.float32),
            'avg_rating': torch.tensor(item_feat.get('avg_rating', 0.0), dtype=torch.float32),
            'revenue': torch.tensor(item_feat.get('revenue', 0.0), dtype=torch.float32),
            'tmdb_genres': torch.tensor(item_feat['tmdb_genres'], dtype=torch.long), # list of ints
            'log_q': torch.tensor(item_feat.get('log_q', 0.0), dtype=torch.float32)
        }
        
        return user_tensor_dict, item_tensor_dict

def create_dataloader(interactions, user_profile, item_profile, batch_size=1024, shuffle=True, num_workers=4):
    dataset = MovielensRecallDataset(interactions, user_profile, item_profile)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
