import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from src.config.settings import CF_TOP_K, CF_WORKERS

# Module-level shared state, inherited by fork workers via COW
_shared = {}

def _compute_item_chunk(chunk_indices):
    """Each worker: R_T[chunk] @ R → normalize → top-K. No full I×I matrix needed."""
    ui_t = _shared['ui_matrix_t']
    ui = _shared['ui_matrix']
    counts = _shared['counts']
    mapping = _shared['mapping']
    top_k = _shared['top_k']

    chunk_co = ui_t[chunk_indices].dot(ui).tocsr()

    result = {}
    for li, gi in enumerate(chunk_indices):
        row = chunk_co.getrow(li)
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

class ItemCFModel:
    def __init__(self, sim_save_path: str):
        self.sim_save_path = Path(sim_save_path)
        self.item_sim_matrix = {}

    def fit(self, train_df: pd.DataFrame, top_k=CF_TOP_K, num_workers=CF_WORKERS):
        train_df = train_df.drop_duplicates(subset=['userId', 'movieId'])
        print(f"ItemCF: Building matrix ({len(train_df)} interactions, {num_workers} workers)...")

        user_ids = train_df['userId'].unique()
        item_ids = train_df['movieId'].unique()
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
        idx_to_item = {i: iid for iid, i in item_to_idx.items()}

        u_idx = train_df['userId'].map(user_to_idx).values
        i_idx = train_df['movieId'].map(item_to_idx).values

        ui_matrix = csr_matrix((np.ones(len(train_df)), (u_idx, i_idx)), shape=(len(user_ids), len(item_ids)))
        ui_matrix_t = ui_matrix.T.tocsr()
        item_counts = np.array(ui_matrix.sum(axis=0)).flatten()

        # Set up shared state before forking
        _shared['ui_matrix'] = ui_matrix
        _shared['ui_matrix_t'] = ui_matrix_t
        _shared['counts'] = item_counts
        _shared['mapping'] = idx_to_item
        _shared['top_k'] = top_k

        print(f"ItemCF: Parallel matmul + prune (Top-{top_k})...")
        chunks = np.array_split(np.arange(len(item_ids)), num_workers)
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(_compute_item_chunk, chunks), total=len(chunks), desc="ItemCF chunks"))

        self.item_sim_matrix = {}
        for res in results:
            self.item_sim_matrix.update(res)
        _shared.clear()

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
