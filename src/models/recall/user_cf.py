import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from src.config.settings import CF_TOP_K, CF_WORKERS

_shared = {}

def _init_worker(matrix, mapping, k):
    _shared['matrix'] = matrix
    _shared['mapping'] = mapping
    _shared['top_k'] = k

def _process_chunk(chunk_indices):
    matrix = _shared['matrix']
    idx_to_user = _shared['mapping']
    top_k = _shared['top_k']
    local_sim = {}
    for i in chunk_indices:
        row = matrix.getrow(i)
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

        local_sim[idx_to_user[i]] = {idx_to_user[idx]: float(s) for idx, s in zip(indices, scores)}
    return local_sim

class UserCFModel:
    def __init__(self, sim_save_path: str):
        self.sim_save_path = Path(sim_save_path)
        self.user_sim_matrix = {}
        self.user_item_dict = {}

    def fit(self, train_df: pd.DataFrame, top_k=CF_TOP_K, num_workers=CF_WORKERS):
        print(f"UserCF: Encoding IDs (Workers: {num_workers})...")
        # Deduplicate interactions
        train_df = train_df.drop_duplicates(subset=['userId', 'movieId'])

        user_ids = train_df['userId'].unique()
        item_ids = train_df['movieId'].unique()
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        idx_to_user = {i: uid for uid, i in user_to_idx.items()}

        self.user_item_dict = train_df.groupby('userId')['movieId'].apply(list).to_dict()
        u_idx = train_df['userId'].map(user_to_idx).values
        item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        i_idx = train_df['movieId'].map(item_to_idx).values

        print("UserCF: Building sparse matrix...")
        ui_matrix = csr_matrix((np.ones(len(train_df)), (u_idx, i_idx)), shape=(len(user_ids), len(item_ids)))

        print("UserCF: Matrix Multiplication (R * R^T)...")
        uu_matrix = ui_matrix.dot(ui_matrix.T)

        print("UserCF: Normalizing...")
        user_counts = np.array(uu_matrix.diagonal()).flatten()
        uu_matrix = uu_matrix.tocoo()
        norm = np.sqrt(user_counts[uu_matrix.row] * user_counts[uu_matrix.col])
        sim_data = uu_matrix.data / (norm + 1e-8)
        refined_matrix = csr_matrix((sim_data, (uu_matrix.row, uu_matrix.col)), shape=uu_matrix.shape)

        print(f"UserCF: Parallel pruning to Top-{top_k}...")
        chunks = np.array_split(np.arange(len(user_ids)), num_workers)

        with Pool(num_workers, initializer=_init_worker, initargs=(refined_matrix, idx_to_user, top_k)) as pool:
            results = list(tqdm(pool.imap(_process_chunk, chunks), total=len(chunks), desc="Pruning"))

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
        print(f"UserCF saved to {self.sim_save_path}")

    def load(self):
        with open(self.sim_save_path, "rb") as f:
            data = pickle.load(f)
            self.user_sim_matrix = data['matrix']
            self.user_item_dict = data['user_item']
        print("UserCF matrix loaded.")
