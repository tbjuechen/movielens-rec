import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

def _process_chunk(chunk_indices, sparse_matrix, idx_to_item, top_k):
    """子进程函数：处理矩阵的一块行"""
    local_sim_matrix = {}
    for i in chunk_indices:
        row = sparse_matrix.getrow(i)
        indices = row.indices
        scores = row.data
        
        mask = indices != i
        indices = indices[mask]
        scores = scores[mask]
        
        if len(scores) == 0:
            continue
            
        if len(scores) > top_k:
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            indices = indices[top_idx]
            scores = scores[top_idx]
        
        orig_item_i = idx_to_item[i]
        local_sim_matrix[orig_item_i] = {idx_to_item[idx]: float(score) for idx, score in zip(indices, scores)}
    return local_sim_matrix

class ItemCFModel:
    def __init__(self, sim_save_path: str):
        self.sim_save_path = Path(sim_save_path)
        self.item_sim_matrix = {}

    def fit(self, train_df: pd.DataFrame, top_k=50, num_workers=None):
        """
        多核并行计算 ItemCF 相似度。
        """
        if num_workers is None:
            num_workers = max(1, cpu_count() - 2)

        print(f"ItemCF: Encoding IDs (Workers: {num_workers})...")
        user_ids = train_df['userId'].unique()
        item_ids = train_df['movieId'].unique()
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        idx_to_item = {i: iid for iid, i in item_to_idx.items()}
        
        u_idx = train_df['userId'].map(user_to_idx).values
        i_idx = train_df['movieId'].map(item_to_idx).values
        
        print("ItemCF: Building User-Item sparse matrix...")
        ui_matrix = csr_matrix((np.ones(len(train_df)), (u_idx, i_idx)), shape=(len(user_ids), len(item_ids)))
        
        print("ItemCF: Matrix Multiplication (R^T * R)...")
        co_matrix = ui_matrix.T.dot(ui_matrix)
        
        print("ItemCF: Normalizing...")
        item_counts = np.array(co_matrix.diagonal()).flatten()
        co_matrix = co_matrix.tocoo()
        row_idx = co_matrix.row
        col_idx = co_matrix.col
        norm = np.sqrt(item_counts[row_idx] * item_counts[col_idx])
        sim_data = co_matrix.data / (norm + 1e-8)
        refined_matrix = csr_matrix((sim_data, (row_idx, col_idx)), shape=co_matrix.shape)
        
        # --- 并行处理 Top-K 截断 ---
        print(f"ItemCF: Parallel pruning to Top-{top_k}...")
        indices = np.arange(len(item_ids))
        chunks = np.array_split(indices, num_workers)
        
        process_func = functools.partial(_process_chunk, sparse_matrix=refined_matrix, idx_to_item=idx_to_item, top_k=top_k)
        
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_func, chunks), total=len(chunks), desc="Pruning Chunks"))
            
        print("ItemCF: Merging results...")
        self.item_sim_matrix = {}
        for res in results:
            self.item_sim_matrix.update(res)
            
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
        print(f"Parallel ItemCF saved to {self.sim_save_path}")

    def load(self):
        with open(self.sim_save_path, "rb") as f:
            self.item_sim_matrix = pickle.load(f)
        print("ItemCF matrix loaded.")
