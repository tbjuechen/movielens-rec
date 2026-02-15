import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import gc
import random

class ShardedListwiseRankingDataset(Dataset):
    """
    流式分片加载器：一次只加载一个特征分片到内存。
    适用于全量 32M 采样后的超大规模数据集。
    """
    def __init__(self, shard_files, item_profile, neg_ratio=9, max_seq_len=20, shuffle_shards=True):
        self.shard_files = shard_files
        self.item_profile = item_profile
        self.neg_ratio = neg_ratio
        self.max_seq_len = max_seq_len
        self.shuffle_shards = shuffle_shards
        
        # 固定 ID 映射
        all_mids = set(self.item_profile['movieId'])
        self.mid_map = {mid: i+1 for i, mid in enumerate(all_mids)}
        self.vocab_size = len(all_mids) + 1
        
        # 预先获取 dense_cols (通过读取第一个分片)
        first_shard = pd.read_parquet(self.shard_files[0])
        cols_to_exclude = ['userId', 'movieId', 'label', 'timestamp', 'click_label', 'seq_history']
        self.dense_cols = [c for c in first_shard.columns if c not in cols_to_exclude and pd.api.types.is_numeric_dtype(first_shard[c])]
        
        # 当前加载的分片数据
        self.current_shard_idx = -1
        self.pos_samples = None
        self.neg_samples = None
        self.shard_size = 0

    def _load_shard(self, shard_idx):
        """加载分片并分离正负样本"""
        if shard_idx == self.current_shard_idx:
            return
        
        # 显式清理旧分片内存
        self.pos_samples = None
        self.neg_samples = None
        gc.collect()
        
        shard_path = self.shard_files[shard_idx]
        logger.info(f"Loading shard: {shard_path.name}")
        df = pd.read_parquet(shard_path)
        
        # 按照 Listwise 逻辑分离正负样本
        self.pos_samples = df[df['label'] == 1].reset_index(drop=True)
        self.neg_samples = df[df['label'] == 0].reset_index(drop=True)
        self.shard_size = len(self.pos_samples)
        self.current_shard_idx = shard_idx
        
        del df
        gc.collect()

    def __len__(self):
        # 这是一个近似值，因为每个分片大小可能微调，
        # 我们假设用户是在外部循环中控制分片切换，或者这里返回总正样本数。
        # 为了配合 PyTorch 标准 DataLoader，我们目前仅支持单个分片的迭代
        # 真正的全量迭代将在训练脚本中通过切换分片实现。
        return self.shard_size

    def _process_item(self, df, idx):
        row = df.iloc[idx]
        mid_idx = self.mid_map.get(row['movieId'], 0)
        
        seq_raw = row.get('seq_history', [])
        seq = seq_raw[-self.max_seq_len:] if isinstance(seq_raw, (list, np.ndarray)) else []
        seq_idx = [self.mid_map.get(m, 0) for m in seq]
        pad_len = self.max_seq_len - len(seq_idx)
        if pad_len > 0: seq_idx = seq_idx + [0] * pad_len
            
        dense_vals = [float(row.get(c, 0.0)) for c in self.dense_cols]
        return mid_idx, seq_idx, dense_vals

    def __getitem__(self, idx):
        # 获取正样本
        pos_mid, pos_seq, pos_dense = self._process_item(self.pos_samples, idx)
        
        # 获取负样本 (1:9)
        neg_mids, neg_seqs, neg_denses = [], [], []
        for i in range(self.neg_ratio):
            neg_idx = idx * self.neg_ratio + i
            n_m, n_s, n_d = self._process_item(self.neg_samples, neg_idx % len(self.neg_samples))
            neg_mids.append(n_m)
            neg_seqs.append(n_s)
            neg_denses.append(n_d)
            
        return {
            'movieId': torch.tensor([pos_mid] + neg_mids, dtype=torch.long),
            'seq_history': torch.tensor([pos_seq] + neg_seqs, dtype=torch.long),
            'dense_feats': torch.tensor([pos_dense] + neg_denses, dtype=torch.float32),
            'click_label': torch.tensor(0, dtype=torch.long)
        }
