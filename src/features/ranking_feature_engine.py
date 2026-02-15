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

    def initialize(self, ref_ratings: pd.DataFrame = None):
        """加载画像表并预计算交叉索引"""
        logger.info("正在初始化精排特征引擎...")
        
        user_p_path = self.ranking_dir / "user_profile_ranking.parquet"
        item_p_path = self.ranking_dir / "item_profile_ranking.parquet"
        
        if not user_p_path.exists() or not item_p_path.exists():
            raise FileNotFoundError("请先运行 scripts/generate_feature_profiles.py 生成画像表！")

        self.user_profile = pd.read_parquet(user_p_path)
        self.item_profile = pd.read_parquet(item_p_path)
        
        # 构建『用户-主创』匹配索引
        logger.info("正在计算『用户-主创』高分匹配索引...")
        # ⚠️ 核心改进：只基于传入的参考数据（训练集）构建偏好索引，防止标签泄露
        ratings = ref_ratings if ref_ratings is not None else pd.read_parquet(self.data_dir / "ratings.parquet")
        
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
        
        # 2.4 语义相似度 (余弦相似度)
        # 逻辑：将用户最近看过的 Last-5 电影的 Embedding 取均值，作为用户的瞬时兴趣向量
        # 然后计算该向量与当前电影 Embedding 的余弦相似度
        logger.info("计算语义相似度特征...")
        
        # 建立 movieId -> embedding 的快速查找字典
        movie_emb_dict = dict(zip(self.item_profile['movieId'], self.item_profile['embedding']))
        
        def compute_semantic_sim(row):
            seq = row.get('seq_history', [])
            target_mid = row['movieId']
            
            if not isinstance(seq, (list, np.ndarray)) or len(seq) == 0:
                return 0.0
            
            target_emb = movie_emb_dict.get(target_mid)
            if target_emb is None or np.sum(np.abs(target_emb)) == 0:
                return 0.0
                
            # 计算历史序列的均值向量
            history_embs = [movie_emb_dict.get(mid) for mid in seq if movie_emb_dict.get(mid) is not None]
            if not history_embs:
                return 0.0
            
            mean_history_emb = np.mean(history_embs, axis=0)
            
            # 余弦相似度
            norm_target = np.linalg.norm(target_emb)
            norm_history = np.linalg.norm(mean_history_emb)
            if norm_target == 0 or norm_history == 0:
                return 0.0
            
            return np.dot(target_emb, mean_history_emb) / (norm_target * norm_history)

        df['semantic_sim'] = df.apply(compute_semantic_sim, axis=1)
        
        logger.success("特征矩阵构建完成。")
        return df
