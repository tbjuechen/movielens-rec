import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from loguru import logger
from tqdm import tqdm

def generate_profiles(ref_ratings=None):
    data_dir = Path("data/processed")
    tmdb_dir = data_dir / "tmdb"
    recall_data_dir = data_dir / "two_tower"
    output_dir = data_dir / "ranking"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始生成精排特征底座 (Feature Profiles)...")

    # --- 1. 加载所有组件 ---
    logger.info("正在加载数据表...")
    # ⚠️ 修正：如果传入了 ref_ratings (训练集)，则基于它计算统计特征
    ratings = ref_ratings if ref_ratings is not None else pd.read_parquet(data_dir / "ratings.parquet")
    movies = pd.read_parquet(data_dir / "movies.parquet")
    
    tmdb_movies = pd.read_parquet(tmdb_dir / "tmdb_movies.parquet")
    tmdb_crew = pd.read_parquet(tmdb_dir / "tmdb_movie_crew.parquet")
    tmdb_cast = pd.read_parquet(tmdb_dir / "tmdb_movie_cast.parquet")
    
    # 加载 Embedding (如果文件还没生成全，先做兼容处理)
    embedding_path = tmdb_dir / "tmdb_embeddings_full.parquet"
    if embedding_path.exists():
        logger.info("检测到语义向量表，正在合并...")
        embeddings = pd.read_parquet(embedding_path)
    else:
        logger.warning("语义向量表尚未生成！将使用空向量占位。")
        embeddings = pd.DataFrame(columns=['movieId', 'embedding'])

    # --- 2. 构建 Item Profile ---
    logger.info("构建 Item Profile...")
    
    # 提取导演 (每部电影一个导演列表)
    directors = tmdb_crew[tmdb_crew['job'] == 'Director'].groupby('movieId')['personId'].apply(list).reset_index(name='director_ids')
    # 提取前 3 名主演
    top_cast = tmdb_cast[tmdb_cast['order'] <= 2].groupby('movieId')['personId'].apply(list).reset_index(name='cast_ids')

    # 基础表
    item_profile = movies[['movieId', 'year', 'genres']].copy()
    
    # 合并 TMDb 业务特征
    item_profile = item_profile.merge(tmdb_movies[['movieId', 'runtime', 'budget', 'revenue', 'vote_average', 'vote_count']], on='movieId', how='left')
    
    # 合并主创 ID
    item_profile = item_profile.merge(directors, on='movieId', how='left')
    item_profile = item_profile.merge(top_cast, on='movieId', how='left')
    
    # 合并语义向量
    item_profile = item_profile.merge(embeddings, on='movieId', how='left')

    # 填充缺失值
    item_profile['director_ids'] = item_profile['director_ids'].apply(lambda x: x if isinstance(x, list) else [])
    item_profile['cast_ids'] = item_profile['cast_ids'].apply(lambda x: x if isinstance(x, list) else [])
    # 缺失的 embedding 用 1024 维零向量填充 (假设维度是 1024)
    sample_emb = embeddings['embedding'].iloc[0] if not embeddings.empty else [0.0] * 1024
    emb_dim = len(sample_emb)
    item_profile['embedding'] = item_profile['embedding'].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else [0.0] * emb_dim)

    # 数值归一化预处理 (针对排序模型)
    item_profile['budget_log'] = np.log1p(item_profile['budget'].fillna(0))
    item_profile['revenue_log'] = np.log1p(item_profile['revenue'].fillna(0))
    
    item_profile.to_parquet(output_dir / "item_profile_ranking.parquet", index=False)
    logger.success(f"Item Profile 已存至 {output_dir / 'item_profile_ranking.parquet'}")

    # --- 3. 构建 User Profile ---
    logger.info("构建 User Profile...")
    
    user_stats = ratings.groupby('userId').agg({
        'rating': ['mean', 'std', 'count']
    })
    user_stats.columns = ['user_avg_rating', 'user_rating_std', 'user_rating_count']
    user_stats = user_stats.reset_index()
    user_stats['user_rating_count_log'] = np.log1p(user_stats['user_rating_count'])
    
    # 获取用户最喜欢的题材 (动态计算)
    logger.info("动态计算用户题材偏好...")
    # 兼容性处理：如果 movies 里没有 genres_idx，使用原始 genres 字段
    genre_col = 'genres_idx' if 'genres_idx' in movies.columns else 'genres'
    user_movie_genres = ratings.merge(movies[['movieId', genre_col]], on='movieId')
    # 展开题材列表
    user_genre_exploded = user_movie_genres.explode(genre_col)
    # 统计每个用户最常看的 Top 3 题材
    user_top_genres = user_genre_exploded.groupby('userId')[genre_col].apply(
        lambda x: x.value_counts().head(3).index.tolist()
    ).reset_index(name='user_top_genres_idx')

    # 合并统计特征与题材偏好
    user_profile = user_stats.merge(user_top_genres, on='userId', how='left')
    
    # 填充缺失题材为空列表
    user_profile['user_top_genres_idx'] = user_profile['user_top_genres_idx'].apply(
        lambda x: x if isinstance(x, (list, np.ndarray)) else []
    )
    
    user_profile.to_parquet(output_dir / "user_profile_ranking.parquet", index=False)
    logger.success(f"User Profile 已基于动态计算存至 {output_dir / 'user_profile_ranking.parquet'}")

    logger.info(f"特征底座构建完成。物品表: {item_profile.shape}, 用户表: {user_profile.shape}")

if __name__ == "__main__":
    generate_profiles()
