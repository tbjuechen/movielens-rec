import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from loguru import logger

def extract_year(title: str) -> int:
    """从标题中提取年份"""
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else 0

class GenreIndexer:
    """题材索引器：将题材字符串映射为数字索引"""
    def __init__(self):
        self.genre_to_idx = {}
        self.idx_to_genre = {}

    def fit(self, genres_series: pd.Series):
        unique_genres = set()
        for g_list in genres_series:
            unique_genres.update(g_list)
        
        # 留出 0 给 Padding 或 Unknown
        self.genre_to_idx = {genre: i + 1 for i, genre in enumerate(sorted(list(unique_genres)))}
        self.idx_to_genre = {i: g for g, i in self.genre_to_idx.items()}
        logger.info(f"GenreIndexer fitted. Found {len(self.genre_to_idx)} unique genres.")

    def transform(self, genres_list: List[str]) -> List[int]:
        return [self.genre_to_idx.get(g, 0) for g in genres_list]

def calculate_item_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    """计算物品侧统计特征：平均分、打分次数（对数化）"""
    stats = df_train.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    stats.columns = ['movieId', 'item_avg_rating', 'item_rating_count']
    
    # 归一化处理
    stats['item_rating_count_log'] = np.log1p(stats['item_rating_count'])
    stats['item_rating_count_norm'] = (stats['item_rating_count_log'] - stats['item_rating_count_log'].min()) / \
                                     (stats['item_rating_count_log'].max() - stats['item_rating_count_log'].min())
    
    # 评分已经在 0.5-5 之间，简单的线性映射到 0-1
    stats['item_avg_rating_norm'] = (stats['item_avg_rating'] - 0.5) / 4.5
    
    return stats[['movieId', 'item_avg_rating_norm', 'item_rating_count_norm']]

def calculate_user_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    """计算用户侧统计特征：平均分、活跃度"""
    stats = df_train.groupby('userId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    stats.columns = ['userId', 'user_avg_rating', 'user_rating_count']
    
    stats['user_rating_count_log'] = np.log1p(stats['user_rating_count'])
    stats['user_rating_count_norm'] = (stats['user_rating_count_log'] - stats['user_rating_count_log'].min()) / \
                                     (stats['user_rating_count_log'].max() - stats['user_rating_count_log'].min())
    
    stats['user_avg_rating_norm'] = (stats['user_avg_rating'] - 0.5) / 4.5
    
    return stats[['userId', 'user_avg_rating_norm', 'user_rating_count_norm']]

def get_user_genre_preference(df_train: pd.DataFrame, df_movies: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """计算用户的题材偏好：根据观影历史提取 Top-K 喜欢的题材"""
    # 合并电影题材
    df_merged = df_train.merge(df_movies[['movieId', 'genres']], on='movieId')
    # 展开题材
    df_exploded = df_merged.explode('genres')
    # 统计用户-题材频次
    user_genre_counts = df_exploded.groupby(['userId', 'genres']).size().reset_index(name='count')
    # 排序并取 Top-K
    user_genre_counts = user_genre_counts.sort_values(['userId', 'count'], ascending=[True, False])
    
    user_top_genres = user_genre_counts.groupby('userId')['genres'].apply(lambda x: list(x)[:top_k]).reset_index()
    user_top_genres.columns = ['userId', 'user_top_genres']
    
    return user_top_genres
