import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from loguru import logger

class ListwiseRankingDataset(Dataset):
    """
    Listwise 加载器：动态支持所有数值特征。
    """
    def __init__(self, samples_df, item_profile, neg_ratio=4, max_seq_len=20):
        # 识别正负样本
        self.pos_samples = samples_df[samples_df['label'] == 1].reset_index(drop=True)
        self.neg_samples = samples_df[samples_df['label'] == 0].reset_index(drop=True)
        
        self.item_profile = item_profile
        self.neg_ratio = neg_ratio
        self.max_seq_len = max_seq_len
        
        # 自动识别数值特征列
        cols_to_exclude = ['userId', 'movieId', 'label', 'timestamp', 'click_label', 'seq_history', 'director_ids', 'cast_ids', 'genres', 'embedding']
        self.dense_cols = [c for c in samples_df.columns if c not in cols_to_exclude and pd.api.types.is_numeric_dtype(samples_df[c])]
        
        # ID 映射
        all_mids = set(self.pos_samples['movieId']) | set(self.item_profile['movieId'])
        self.mid_map = {mid: i+1 for i, mid in enumerate(all_mids)}
        self.vocab_size = len(all_mids) + 1
        
        logger.info(f"ListwiseDataset ready. Dense Features: {len(self.dense_cols)}")

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
            
        # 动态提取所有特征列
        dense_vals = [float(row.get(c, 0.0)) for c in self.dense_cols]
        
        return mid_idx, seq_idx, dense_vals

    def __getitem__(self, idx):
        pos_row = self.pos_samples.iloc[idx]
        pos_mid, pos_seq, pos_dense = self._process_item(pos_row)
        
        neg_mids, neg_seqs, neg_denses = [], [], []
        for i in range(self.neg_ratio):
            neg_idx = idx * self.neg_ratio + i
            neg_row = self.neg_samples.iloc[neg_idx % len(self.neg_samples)]
            m, s, d = self._process_item(neg_row)
            neg_mids.append(m)
            neg_seqs.append(s)
            neg_denses.append(d)
            
        return {
            'movieId': torch.tensor([pos_mid] + neg_mids, dtype=torch.long),
            'seq_history': torch.tensor([pos_seq] + neg_seqs, dtype=torch.long),
            'dense_feats': torch.tensor([pos_dense] + neg_denses, dtype=torch.float32),
            'click_label': torch.tensor(0, dtype=torch.long)
        }
