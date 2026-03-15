import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import functools

from src.config.settings import CF_TOP_K, CF_WORKERS

# Module-level shared state for worker processes (inherited via fork COW)
_shared = {}

def _init_worker(matrix, mapping, k):
    _shared['matrix'] = matrix
    _shared['mapping'] = mapping
    _shared['top_k'] = k

def _process_chunk(chunk_indices):
    matrix = _shared['matrix']
    idx_to_item = _shared['mapping']
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

        local_sim[idx_to_item[i]] = {idx_to_item[idx]: float(s) for idx, s in zip(indices, scores)}
    return local_sim

class ItemCFModel:
    def __init__(self, sim_save_path: str):
        self.sim_save_path = Path(sim_save_path)
        self.item_sim_matrix = {}

    def fit(self, train_df: pd.DataFrame, top_k=CF_TOP_K, num_workers=CF_WORKERS):
        print(f"ItemCF: Encoding IDs (Workers: {num_workers})...")
        # Deduplicate interactions
        train_df = train_df.drop_duplicates(subset=['userId', 'movieId'])

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
        norm = np.sqrt(item_counts[co_matrix.row] * item_counts[co_matrix.col])
        sim_data = co_matrix.data / (norm + 1e-8)
        refined_matrix = csr_matrix((sim_data, (co_matrix.row, co_matrix.col)), shape=co_matrix.shape)

        print(f"ItemCF: Parallel pruning to Top-{top_k}...")
        chunks = np.array_split(np.arange(len(item_ids)), num_workers)

        with Pool(num_workers, initializer=_init_worker, initargs=(refined_matrix, idx_to_item, top_k)) as pool:
            results = list(tqdm(pool.imap(_process_chunk, chunks), total=len(chunks), desc="Pruning"))

        self.item_sim_matrix = {}
        for res in results:
            self.item_sim_matrix.update(res)

        self.save()

    def retrieve(self, user_history: list, k=50):
        interacted = set(user_history)
        rank = {}
        for item in user_history:
            if item not in self.item_sim_matrix:
                continue
            for related_item, score in self.item_sim_matrix[item].items():
                if related_item in interacted:
                    continue
                rank[related_item] = rank.get(related_item, 0) + score
        sorted_res = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:k]
        return [res[0] for res in sorted_res]

    def save(self):
        self.sim_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sim_save_path, "wb") as f:
            pickle.dump(self.item_sim_matrix, f)
        print(f"ItemCF saved to {self.sim_save_path}")

    def load(self):
        with open(self.sim_save_path, "rb") as f:
            self.item_sim_matrix = pickle.load(f)
        print("ItemCF matrix loaded.")
