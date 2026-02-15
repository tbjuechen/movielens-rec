import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from loguru import logger

class ListwiseRankingDataset(Dataset):
    """
    Listwise 加载器：将 1 个正样本和 N 个负样本打包为一个 Batch 单元。
    支持多任务标签 (Click + Rating)。
    """
    def __init__(self, samples_df, item_profile, neg_ratio=4, max_seq_len=20):
        # 1. 识别正样本
        self.pos_samples = samples_df[samples_df['label'] == 1].reset_index(drop=True)
        # 2. 识别负样本
        self.neg_samples = samples_df[samples_df['label'] == 0].reset_index(drop=True)
        
        self.item_profile = item_profile
        self.neg_ratio = neg_ratio
        self.max_seq_len = max_seq_len
        
        # ID 映射
        all_mids = set(self.pos_samples['movieId']) | set(self.item_profile['movieId'])
        self.mid_map = {mid: i+1 for i, mid in enumerate(all_mids)}
        self.vocab_size = len(all_mids) + 1

    def __len__(self):
        return len(self.pos_samples)

    def _process_item(self, row):
        mid_idx = self.mid_map.get(row['movieId'], 0)
        
        seq_raw = row.get('seq_history', [])
        seq = seq_raw[-self.max_seq_len:] if isinstance(seq_raw, (list, np.ndarray)) else []
        seq_idx = [self.mid_map.get(m, 0) for m in seq]
        pad_len = self.max_seq_len - len(seq_idx)
        if pad_len > 0:
            seq_idx = seq_idx + [0] * pad_len
            
        dense_cols = ['user_avg_rating', 'vote_average', 'semantic_sim', 'genre_match_score']
        dense_vals = [float(row.get(c, 0.0)) for c in dense_cols]
        
        # 获取原始评分并归一化到 [0, 1]，防止 MSE 梯度爆炸
        rating = float(row.get('rating', 0.0)) / 5.0
        
        return mid_idx, seq_idx, dense_vals, rating

    def __getitem__(self, idx):
        # 1. 获取正样本
        pos_row = self.pos_samples.iloc[idx]
        pos_mid, pos_seq, pos_dense, pos_rating = self._process_item(pos_row)
        
        # 2. 获取对应的 N 个负样本
        neg_mids, neg_seqs, neg_denses, neg_ratings = [], [], [], []
        for i in range(self.neg_ratio):
            neg_idx = idx * self.neg_ratio + i
            neg_row = self.neg_samples.iloc[neg_idx % len(self.neg_samples)]
            m, s, d, r = self._process_item(neg_row)
            neg_mids.append(m)
            neg_seqs.append(s)
            neg_denses.append(d)
            neg_ratings.append(r)
            
        return {
            'movieId': torch.tensor([pos_mid] + neg_mids, dtype=torch.long),
            'seq_history': torch.tensor([pos_seq] + neg_seqs, dtype=torch.long),
            'dense_feats': torch.tensor([pos_dense] + neg_denses, dtype=torch.float32),
            'click_label': torch.tensor(0, dtype=torch.long), # 正样本在 0 位
            'rating_label': torch.tensor([pos_rating] + neg_ratings, dtype=torch.float32)
        }
