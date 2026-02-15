import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from loguru import logger

class RankingDataset(Dataset):
    def __init__(self, samples_df, item_profile, max_seq_len=20):
        self.samples = samples_df
        self.item_profile = item_profile
        self.max_seq_len = max_seq_len
        
        # 预处理：构建 movieId -> 连续索引的映射
        # 注意：真实项目中这个 map 应该持久化，这里简化处理
        all_mids = set(self.samples['movieId']) | set(self.item_profile['movieId'])
        # 0 is reserved for padding
        self.mid_map = {mid: i+1 for i, mid in enumerate(all_mids)}
        self.vocab_size = len(all_mids) + 1
        
        logger.info(f"Dataset initialized. Vocab Size: {self.vocab_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        
        # 1. Sparse Features (ID)
        mid_idx = self.mid_map.get(row['movieId'], 0)
        
        # 2. Sequence Features (History)
        seq_raw = row.get('seq_history', [])
        if not isinstance(seq_raw, (list, np.ndarray)):
            seq_raw = []
        
        # 截断与填充
        seq = seq_raw[-self.max_seq_len:] # 取最近 N 个
        seq_idx = [self.mid_map.get(m, 0) for m in seq]
        pad_len = self.max_seq_len - len(seq_idx)
        if pad_len > 0:
            seq_idx = seq_idx + [0] * pad_len
            
        # 3. Dense Features (需要从 RankingFeatureEngine 的输出里拿)
        # 这里为了演示，我们假设输入的 samples_df 已经是通过 FeatureEngine 处理过的大宽表
        # 提取几个核心数值特征
        dense_cols = ['user_avg_rating', 'vote_average', 'semantic_sim', 'genre_match_score']
        dense_vals = [float(row.get(c, 0.0)) for c in dense_cols]

        return {
            'movieId': torch.tensor(mid_idx, dtype=torch.long),
            'seq_history': torch.tensor(seq_idx, dtype=torch.long),
            'dense_feats': torch.tensor(dense_vals, dtype=torch.float32),
            'label': torch.tensor(float(row['label']), dtype=torch.float32)
        }
