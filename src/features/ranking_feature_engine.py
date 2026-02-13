import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Dict, Set

class RankingFeatureEngine:
    """
    精排特征交叉引擎：负责生成 User-Item 匹配特征。
    """
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.ranking_dir = self.data_dir / "ranking"
        self.tmdb_dir = self.data_dir / "tmdb"
        
        # 特征查找表
        self.user_profile = None
        self.item_profile = None
        
        # 交叉匹配辅助索引 (userId -> Set of personIds)
        self.user_high_score_directors: Dict[int, Set[int]] = {}
        self.user_high_score_actors: Dict[int, Set[int]] = {}

    def initialize(self):
        """加载画像表并预计算交叉索引"""
        logger.info("正在初始化精排特征引擎...")
        
        user_p_path = self.ranking_dir / "user_profile_ranking.parquet"
        item_p_path = self.ranking_dir / "item_profile_ranking.parquet"
        
        if not user_p_path.exists() or not item_p_path.exists():
            raise FileNotFoundError("请先运行 scripts/generate_feature_profiles.py 生成画像表！")

        self.user_profile = pd.read_parquet(user_p_path)
        self.item_profile = pd.read_parquet(item_p_path)
        
        # 构建『用户-主创』匹配索引
        logger.info("正在计算『用户-主创』高分匹配索引 (Label > 4.0)...")
        ratings = pd.read_parquet(self.data_dir / "ratings.parquet")
        tmdb_crew = pd.read_parquet(self.tmdb_dir / "tmdb_movie_crew.parquet")
        tmdb_cast = pd.read_parquet(self.tmdb_dir / "tmdb_movie_cast.parquet")
        
        # 筛选用户打过高分的电影
        high_score_mids = ratings[ratings['rating'] >= 4.0][['userId', 'movieId']]
        
        # 用户 -> 喜欢的导演集合
        user_dir_df = high_score_mids.merge(tmdb_crew[tmdb_crew['job'] == 'Director'], on='movieId')
        self.user_high_score_directors = user_dir_df.groupby('userId')['personId'].apply(set).to_dict()
        
        # 用户 -> 喜欢的演员集合 (只取前3主演)
        user_act_df = high_score_mids.merge(tmdb_cast[tmdb_cast['order'] <= 2], on='movieId')
        self.user_high_score_actors = user_act_df.groupby('userId')['personId'].apply(set).to_dict()
        
        logger.success("精排特征引擎初始化完成。")

    def build_feature_matrix(self, samples_df: pd.DataFrame) -> pd.DataFrame:
        """
        将样本集(userId, movieId, label)转化为特征矩阵。
        """
        logger.info(f"开始为 {len(samples_df):,} 条样本提取特征...")
        
        # 1. 基础画像拼接
        df = samples_df.merge(self.user_profile, on='userId', how='left')
        df = df.merge(self.item_profile, on='movieId', how='left')
        
        # 2. 交叉匹配特征 (动态计算)
        # 2.1 导演匹配
        def check_director_match(row):
            uid, directors = row['userId'], row['director_ids']
            if uid in self.user_high_score_directors and isinstance(directors, (list, np.ndarray)):
                return 1 if any(d in self.user_high_score_directors[uid] for d in directors) else 0
            return 0
        
        # 2.2 演员匹配计数
        def count_actor_match(row):
            uid, actors = row['userId'], row['cast_ids']
            if uid in self.user_high_score_actors and isinstance(actors, (list, np.ndarray)):
                return sum(1 for a in actors if a in self.user_high_score_actors[uid])
            return 0

        logger.info("计算主创匹配特征...")
        df['is_director_match'] = df.apply(check_director_match, axis=1)
        df['actor_match_count'] = df.apply(count_actor_match, axis=1)
        
        # 2.3 评分差值
        df['rating_diff'] = df['vote_average'] - df['user_avg_rating']
        
        logger.success("特征矩阵构建完成。")
        return df
