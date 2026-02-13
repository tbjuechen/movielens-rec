import pandas as pd
import pickle
from typing import List, Tuple
from pathlib import Path
from loguru import logger
from src.models.recall.base import BaseRecall

class PopularityRecall(BaseRecall):
    """
    热门召回：根据电影的全局评分次数（流行度）进行推荐。
    """
    def __init__(self):
        super().__init__(name="PopularityRecall")
        self.top_movies: List[Tuple[int, float]] = []

    def train(self, df_train: pd.DataFrame, **kwargs) -> None:
        """
        统计每部电影的评分总数作为流行度分数。
        """
        logger.info("Training PopularityRecall: calculating movie counts...")
        
        # 统计 movieId 出现的次数
        movie_counts = df_train.groupby('movieId').size().reset_index(name='score')
        
        # 按分数（流行度）降序排列
        movie_counts = movie_counts.sort_values(by='score', ascending=False)
        
        # 转换为列表形式 [(movieId, score), ...]
        self.top_movies = list(zip(movie_counts['movieId'], movie_counts['score']))
        logger.info(f"PopularityRecall trained. Found {len(self.top_movies)} movies.")

    def recall(self, user_id: int, top_n: int = 100) -> List[Tuple[int, float]]:
        """
        返回全站最热门的前 top_n 部电影。
        """
        return self.top_movies[:top_n]

    def save(self, path: str) -> None:
        """将热门列表持久化为 pickle 文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.top_movies, f)
        logger.info(f"PopularityRecall saved to {path}")

    def load(self, path: str) -> None:
        """从 pickle 文件加载热门列表"""
        with open(path, 'rb') as f:
            self.top_movies = pickle.load(f)
        logger.info(f"PopularityRecall loaded from {path}")
