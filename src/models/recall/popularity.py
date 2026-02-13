import pandas as pd
import pickle
from typing import List, Tuple, Dict
from pathlib import Path
from loguru import logger
from src.models.recall.base import BaseRecall

class PopularityRecall(BaseRecall):
    """
    热门召回：根据电影的全局评分次数（流行度）进行推荐，并过滤掉用户已看过的电影。
    """
    def __init__(self):
        super().__init__(name="PopularityRecall")
        self.top_movies: List[Tuple[int, float]] = []
        self.user_history: Dict[int, set] = {}

    def train(self, df_train: pd.DataFrame, **kwargs) -> None:
        """
        统计每部电影的评分总数作为流行度分数，并记录用户观影历史。
        """
        logger.info("Training PopularityRecall: calculating movie counts and user history...")
        
        # 1. 统计 movieId 出现的次数
        movie_counts = df_train.groupby('movieId').size().reset_index(name='score')
        movie_counts = movie_counts.sort_values(by='score', ascending=False)
        self.top_movies = list(zip(movie_counts['movieId'], movie_counts['score'].astype(float)))
        
        # 2. 记录用户看过的电影集合
        self.user_history = df_train.groupby('userId')['movieId'].apply(set).to_dict()
        
        logger.info(f"PopularityRecall trained. Movies: {len(self.top_movies)}, Users: {len(self.user_history)}")

    def recall(self, user_id: int, top_n: int = 100) -> List[Tuple[int, float]]:
        """
        返回全站最热门且用户未看过的前 top_n 部电影。
        """
        history = self.user_history.get(user_id, set())
        results = []
        
        for mid, score in self.top_movies:
            if mid not in history:
                results.append((mid, score))
            if len(results) >= top_n:
                break
        
        return results

    def save(self, path: str) -> None:
        """将热门列表和历史记录持久化"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'top_movies': self.top_movies,
            'user_history': self.user_history
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"PopularityRecall saved to {path}")

    def load(self, path: str) -> None:
        """加载数据"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.top_movies = data['top_movies']
            self.user_history = data['user_history']
        logger.info(f"PopularityRecall loaded from {path}")
