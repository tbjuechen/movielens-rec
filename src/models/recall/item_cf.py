import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from pathlib import Path
import math

class ItemCFModel:
    def __init__(self, sim_save_path: str):
        self.sim_save_path = Path(sim_save_path)
        self.item_sim_matrix = {}

    def fit(self, train_df: pd.DataFrame, top_k=50):
        """
        使用稀疏矩阵加速计算 ItemCF 相似度。
        """
        print("ItemCF: Encoding IDs for sparse matrix...")
        # 1. 局部编码 (为了矩阵对齐)
        user_ids = train_df['userId'].unique()
        item_ids = train_df['movieId'].unique()
        
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        idx_to_item = {i: iid for iid, i in item_to_idx.items()}
        
        u_idx = train_df['userId'].map(user_to_idx).values
        i_idx = train_df['movieId'].map(item_to_idx).values
        
        # 2. 构建 User-Item 稀疏矩阵
        # 值为 1 代表交互过
        print("ItemCF: Building User-Item sparse matrix...")
        rows = u_idx
        cols = i_idx
        data = np.ones(len(train_df))
        ui_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
        
        # 3. 计算物品共现矩阵 C = R^T * R
        print("ItemCF: Calculating co-occurrence matrix (Matrix Multiplication)...")
        # 这一步是核心，利用稀疏矩阵乘法计算所有物品对的共现次数
        co_matrix = ui_matrix.T.dot(ui_matrix)
        
        # 4. 计算余弦相似度
        print("ItemCF: Normalizing similarity scores...")
        # 物品出现次数 (对角线元素)
        item_counts = np.array(co_matrix.diagonal()).flatten()
        
        # 为了避免循环，我们直接操作稀疏矩阵的数据
        # W_ij = C_ij / sqrt(count_i * count_j)
        co_matrix = co_matrix.tocoo()
        row_idx = co_matrix.row
        col_idx = co_matrix.col
        sim_data = co_matrix.data
        
        # 向量化计算相似度
        norm = np.sqrt(item_counts[row_idx] * item_counts[col_idx])
        sim_data = sim_data / (norm + 1e-8)
        
        # 5. 构建最终字典并截断 Top-K
        print(f"ItemCF: Pruning to Top-{top_k} per item...")
        final_sim_matrix = {}
        
        # 转回 CSR 方便按行取 Top-K
        refined_matrix = csr_matrix((sim_data, (row_idx, col_idx)), shape=co_matrix.shape)
        
        for i in range(len(item_ids)):
            row = refined_matrix.getrow(i)
            # 获取该行所有非零列的索引和分值
            indices = row.indices
            scores = row.data
            
            # 排除自身
            mask = indices != i
            indices = indices[mask]
            scores = scores[mask]
            
            if len(scores) == 0:
                continue
                
            # 取 Top-K
            if len(scores) > top_k:
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                indices = indices[top_idx]
                scores = scores[top_idx]
            
            # 存回原始 movieId
            orig_item_i = idx_to_item[i]
            final_sim_matrix[orig_item_i] = {idx_to_item[idx]: float(score) for idx, score in zip(indices, scores)}
            
        self.item_sim_matrix = final_sim_matrix
        self.save()

    def retrieve(self, user_history: list, k=50):
        rank = {}
        for item in user_history:
            if item not in self.item_sim_matrix:
                continue
            for related_item, score in self.item_sim_matrix[item].items():
                if related_item in user_history:
                    continue
                rank[related_item] = rank.get(related_item, 0) + score
        
        sorted_res = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:k]
        return [res[0] for res in sorted_res]

    def save(self):
        self.sim_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sim_save_path, "wb") as f:
            pickle.dump(self.item_sim_matrix, f)
        print(f"High-performance ItemCF saved to {self.sim_save_path}")

    def load(self):
        with open(self.sim_save_path, "rb") as f:
            self.item_sim_matrix = pickle.load(f)
        print("ItemCF matrix loaded.")
