import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from src.config.settings import CF_TOP_K, CF_WORKERS

_shared = {}

def _compute_user_chunk(chunk_indices):
    """Each worker: R[chunk] @ R^T → normalize → top-K. No full U×U matrix."""
    ui = _shared['ui_matrix']
    ui_t = _shared['ui_matrix_t']
    counts = _shared['counts']
    mapping = _shared['mapping']
    top_k = _shared['top_k']

    chunk_uu = ui[chunk_indices].dot(ui_t).tocsr()

    result = {}
    for li, gi in enumerate(chunk_indices):
        row = chunk_uu.getrow(li)
        idx, data = row.indices, row.data

        mask = idx != gi
        idx, data = idx[mask], data[mask]
        if len(data) == 0:
            continue

        norm = np.sqrt(counts[gi] * counts[idx])
        scores = data / (norm + 1e-8)

        if len(scores) > top_k:
            top = np.argpartition(scores, -top_k)[-top_k:]
            idx, scores = idx[top], scores[top]

        result[mapping[gi]] = {mapping[j]: float(s) for j, s in zip(idx, scores)}
    return result

class UserCFModel:
    def __init__(self, sim_save_path: str):
        self.sim_save_path = Path(sim_save_path)
        self.user_sim_matrix = {}
        self.user_item_dict = {}

    def fit(self, train_df: pd.DataFrame, top_k=CF_TOP_K, num_workers=CF_WORKERS):
        train_df = train_df.drop_duplicates(subset=['userId', 'movieId'])
        print(f"UserCF: Building matrix ({len(train_df)} interactions, {num_workers} workers)...")

        user_ids = train_df['userId'].unique()
        item_ids = train_df['movieId'].unique()
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        idx_to_user = {i: uid for uid, i in user_to_idx.items()}
        item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

        self.user_item_dict = train_df.groupby('userId')['movieId'].apply(list).to_dict()
        u_idx = train_df['userId'].map(user_to_idx).values
        i_idx = train_df['movieId'].map(item_to_idx).values

        ui_matrix = csr_matrix((np.ones(len(train_df)), (u_idx, i_idx)), shape=(len(user_ids), len(item_ids)))
        ui_matrix_t = ui_matrix.T.tocsr()
        user_counts = np.array(ui_matrix.sum(axis=1)).flatten()

        _shared['ui_matrix'] = ui_matrix
        _shared['ui_matrix_t'] = ui_matrix_t
        _shared['counts'] = user_counts
        _shared['mapping'] = idx_to_user
        _shared['top_k'] = top_k

        print(f"UserCF: Parallel matmul + prune (Top-{top_k})...")
        chunks = np.array_split(np.arange(len(user_ids)), num_workers)
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(_compute_user_chunk, chunks), total=len(chunks), desc="UserCF chunks"))

        self.user_sim_matrix = {}
        for res in results:
            self.user_sim_matrix.update(res)
        _shared.clear()

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
        print(f"UserCF saved to {self.sim_save_path}")

    def load(self):
        with open(self.sim_save_path, "rb") as f:
            data = pickle.load(f)
            self.user_sim_matrix = data['matrix']
            self.user_item_dict = data['user_item']
        print("UserCF matrix loaded.")
