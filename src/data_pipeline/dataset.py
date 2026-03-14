import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MovielensRecallDataset(Dataset):
    def __init__(self, interactions_df: pd.DataFrame, user_profile_df: pd.DataFrame, item_profile_df: pd.DataFrame):
        self.interactions = interactions_df.reset_index(drop=True)
        # 优化：不使用 to_dict，直接使用索引加速
        self.user_profile = user_profile_df.set_index('userId')
        self.item_profile = item_profile_df.set_index('movieId')

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = int(row['userId'])
        item_id = int(row['movieId'])
        
        # 直接从 DataFrame 索引中通过 .loc 提取
        user_feat = self.user_profile.loc[user_id]
        
        # 优化：检查 item_id 是否在索引中，不在则返回默认全零特征
        if item_id in self.item_profile.index:
            item_feat = self.item_profile.loc[item_id]
        else:
            # Fallback for items missing from profile (cold start or filter issue)
            item_feat = {
                'release_year_val': 0.0,
                'avg_rating': 0.0,
                'revenue': 0.0,
                'tmdb_genres': [0] * 5,
                'log_q': np.log(1e-10) # 对应 Log-Q 默认极小值
            }
        
        user_tensor_dict = {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'avg_rating': torch.tensor(user_feat['avg_rating'], dtype=torch.float32),
            'activity': torch.tensor(user_feat['activity'], dtype=torch.float32),
            'history': torch.tensor(user_feat['history'], dtype=torch.long),
            'history_ts_diff': torch.tensor(user_feat['history_ts_diff'], dtype=torch.float32),
            'top_genres': torch.tensor(user_feat['top_genres'], dtype=torch.long)
        }
        
        item_tensor_dict = {
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'release_year': torch.tensor(item_feat['release_year_val'], dtype=torch.float32), # 对应 03 里的新列名
            'avg_rating': torch.tensor(item_feat['avg_rating'], dtype=torch.float32),
            'revenue': torch.tensor(item_feat['revenue'], dtype=torch.float32),
            'tmdb_genres': torch.tensor(item_feat['tmdb_genres'], dtype=torch.long),
            'log_q': torch.tensor(item_feat['log_q'], dtype=torch.float32)
        }
        
        return user_tensor_dict, item_tensor_dict

def create_dataloader(interactions, user_profile, item_profile, batch_size=1024, shuffle=True, num_workers=4):
    dataset = MovielensRecallDataset(interactions, user_profile, item_profile)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
