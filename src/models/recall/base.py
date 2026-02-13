from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import pandas as pd
from loguru import logger

class BaseRecall(ABC):
    """
    推荐系统召回模块基类。
    所有具体的召回策略（如热门召回、ItemCF、向量化召回）都应继承此类。
    """
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initializing Recall Model: {self.name}")

    @abstractmethod
    def train(self, df_train: pd.DataFrame, **kwargs) -> None:
        """
        离线训练接口：用于计算索引、训练模型或生成统计表。
        """
        pass

    @abstractmethod
    def recall(self, user_id: int, top_n: int = 100) -> List[Tuple[int, float]]:
        """
        在线召回接口：为指定用户返回候选电影列表。
        返回格式: [(movieId, score), ...]
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型或中间统计结果"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """加载模型或中间统计结果"""
        pass
