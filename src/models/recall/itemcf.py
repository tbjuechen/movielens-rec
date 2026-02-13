import pandas as pd
import numpy as np
import pickle
from typing import List, Tuple, Dict
from pathlib import Path
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from src.models.recall.base import BaseRecall

class ItemCFRecall(BaseRecall):
    """
    ItemCF 召回：基于物品相似度进行个性化推荐。
    """
    def __init__(self, top_k: int = 50):
        super().__init__(name="ItemCFRecall")
        self.top_k = top_k  # 每部电影只保留最相似的 K 部
        self.sim_matrix: Dict[int, List[Tuple[int, float]]] = {}
        self.user_history: Dict[int, List[int]] = {}

    def train(self, df_train: pd.DataFrame, **kwargs) -> None:
        """
        计算电影之间的相似度矩阵。
        """
        logger.info("Training ItemCFRecall: building user-item matrix...")
        
        # 1. 过滤掉评分较少的电影和用户，以减小计算量并提升质量
        # 这里可以根据 EDA 的结果进行调整
        min_movie_ratings = 10
        movie_counts = df_train.groupby('movieId').size()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df_filtered = df_train[df_train['movieId'].isin(valid_movies)]

        # 2. 构建编码映射（movieId -> matrix_index）
        unique_users = df_filtered['userId'].unique()
        unique_movies = df_filtered['movieId'].unique()
        
        user_map = {uid: i for i, uid in enumerate(unique_users)}
        movie_map = {mid: i for i, mid in enumerate(unique_movies)}
        inv_movie_map = {v: k for k, v in movie_map.items()}

        # 3. 构建稀疏矩阵
        rows = df_filtered['userId'].map(user_map).values
        cols = df_filtered['movieId'].map(movie_map).values
        data = df_filtered['rating'].values
        
        user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_movies)))
        
        # 4. 计算物品余弦相似度（转置矩阵计算 item-item）
        logger.info("Calculating cosine similarity (this may take a while)...")
        item_sim = cosine_similarity(user_item_matrix.T, dense_output=False)
        
        # 5. 只保留 Top-K 相似项以节省内存
        logger.info(f"Pruning similarity matrix (top_k={self.top_k})...")
        self.sim_matrix = {}
        for i in range(item_sim.shape[0]):
            row = item_sim.getrow(i)
            # 获取该行所有非零元素的索引和值
            indices = row.indices
            data = row.data
            
            # 排序并取 Top-K
            best_idx = np.argsort(data)[-(self.top_k + 1):-1][::-1] # 排除自身
            mid = inv_movie_map[i]
            self.sim_matrix[mid] = [(inv_movie_map[indices[idx]], data[idx]) for idx in best_idx]

        # 6. 保存用户历史，用于在线召回
        logger.info("Building user history cache...")
        self.user_history = df_filtered.groupby('userId')['movieId'].apply(list).to_dict()

    def recall(self, user_id: int, top_n: int = 100) -> List[Tuple[int, float]]:
        """
        根据用户历史看过的电影，推荐相似电影。
        """
        if user_id not in self.user_history:
            return []
        
        history = self.user_history[user_id]
        scores = {}
        
        for mid in history:
            if mid not in self.sim_matrix:
                continue
            for sim_mid, sim_score in self.sim_matrix[mid]:
                if sim_mid in history:
                    continue
                scores[sim_mid] = scores.get(sim_mid, 0) + sim_score
        
        # 按得分排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]

    def save(self, path: str) -> None:
        data = {
            'sim_matrix': self.sim_matrix,
            'user_history': self.user_history
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"ItemCFRecall saved to {path}")

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.sim_matrix = data['sim_matrix']
            self.user_history = data['user_history']
        logger.info(f"ItemCFRecall loaded from {path}")
