import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

def _process_user_chunk(chunk_indices, sparse_matrix, idx_to_user, top_k):
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
        orig_user_i = idx_to_user[i]
        local_sim_matrix[orig_user_i] = {idx_to_user[idx]: float(score) for idx, score in zip(indices, scores)}
    return local_sim_matrix

class UserCFModel:
    def __init__(self, sim_save_path: str):
        self.sim_save_path = Path(sim_save_path)
        self.user_sim_matrix = {}
        self.user_item_dict = {}

    def fit(self, train_df: pd.DataFrame, top_k=50, num_workers=None):
        """
        多核并行计算 UserCF 相似度。
        """
        if num_workers is None:
            num_workers = max(1, cpu_count() - 2)

        print(f"UserCF: Encoding IDs (Workers: {num_workers})...")
        user_ids = train_df['userId'].unique()
        item_ids = train_df['movieId'].unique()
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        idx_to_user = {i: uid for uid, i in user_to_idx.items()}
        
        self.user_item_dict = train_df.groupby('userId')['movieId'].apply(list).to_dict()
        u_idx = train_df['userId'].map(user_to_idx).values
        i_idx = train_df['movieId'].map(item_to_idx).values
        
        print("UserCF: Building sparse matrix...")
        ui_matrix = csr_matrix((np.ones(len(train_df)), (u_idx, i_idx)), shape=(len(user_ids), len(item_ids)))
        
        print("UserCF: Matrix Multiplication (R * R^T)...")
        uu_matrix = ui_matrix.dot(ui_matrix.T)
        
        print("UserCF: Normalizing...")
        user_counts = np.array(uu_matrix.diagonal()).flatten()
        uu_matrix = uu_matrix.tocoo()
        row_idx = uu_matrix.row
        col_idx = uu_matrix.col
        norm = np.sqrt(user_counts[row_idx] * user_counts[col_idx])
        sim_data = uu_matrix.data / (norm + 1e-8)
        refined_matrix = csr_matrix((sim_data, (row_idx, col_idx)), shape=uu_matrix.shape)
        
        # --- 并行处理 Top-K 截断 ---
        print(f"UserCF: Parallel pruning to Top-{top_k}...")
        indices = np.arange(len(user_ids))
        chunks = np.array_split(indices, num_workers)
        
        process_func = functools.partial(_process_user_chunk, sparse_matrix=refined_matrix, idx_to_user=idx_to_user, top_k=top_k)
        
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_func, chunks), total=len(chunks), desc="Pruning User Chunks"))
            
        print("UserCF: Merging results...")
        self.user_sim_matrix = {}
        for res in results:
            self.user_sim_matrix.update(res)
            
        self.save()

    def retrieve(self, user_id: int, k=50):
        if user_id not in self.user_sim_matrix:
            return []
        rank = {}
        interacted_items = set(self.user_item_dict.get(user_id, []))
        for v, sim in self.user_sim_matrix[user_id].items():
            for item in self.user_item_dict.get(v, []):
                if item in interacted_items:
                    continue
                rank[item] = rank.get(item, 0) + sim
        sorted_res = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:k]
        return [res[0] for res in sorted_res]

    def save(self):
        self.sim_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sim_save_path, "wb") as f:
            data = {'matrix': self.user_sim_matrix, 'user_item': self.user_item_dict}
            pickle.dump(data, f)
        print(f"Parallel UserCF saved to {self.sim_save_path}")

    def load(self):
        with open(self.sim_save_path, "rb") as f:
            data = pickle.load(f)
            self.user_sim_matrix = data['matrix']
            self.user_item_dict = data['user_item']
        print("UserCF matrix loaded.")
