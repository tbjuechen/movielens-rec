import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import pickle
from src.preprocessing import (
    GenreIndexer, 
    calculate_item_stats, 
    calculate_user_stats, 
    get_user_genre_preference,
    extract_year
)

def prepare_data_v2():
    ratings_path = Path("data/processed/ratings.parquet")
    movies_path = Path("data/processed/movies.parquet")
    output_dir = Path("data/processed/two_tower")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ratings_path.exists() or not movies_path.exists():
        logger.error("Required parquet files not found in data/processed/")
        return

    # --- 1. 加载数据 ---
    logger.info("Loading data...")
    df_ratings = pd.read_parquet(ratings_path)
    df_movies = pd.read_parquet(movies_path)
    
    # --- 2. 时间切分 (80/10/10) ---
    logger.info("Sorting and splitting data...")
    df_ratings = df_ratings.sort_values(by="timestamp").reset_index(drop=True)
    n = len(df_ratings)
    df_train = df_ratings.iloc[:int(n * 0.8)].copy()
    df_val = df_ratings.iloc[int(n * 0.8):int(n * 0.9)].copy()
    df_test = df_ratings.iloc[int(n * 0.9):].copy()

    # --- 3. ID 映射 (仅基于训练集) ---
    unique_users = df_train['userId'].unique()
    unique_movies = df_train['movieId'].unique()
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    movie_map = {mid: i for i, mid in enumerate(unique_movies)}
    
    # --- 4. 物品侧特征加工 (Item Features) ---
    logger.info("Processing item features...")
    # 题材索引
    gi = GenreIndexer()
    gi.fit(df_movies['genres'])
    df_movies['genres_idx'] = df_movies['genres'].apply(gi.transform)
    
    # 统计特征
    df_item_stats = calculate_item_stats(df_train)
    
    # 年份特征归一化
    df_movies['release_year'] = df_movies['title'].apply(extract_year)
    min_y, max_y = df_movies['release_year'].replace(0, 2000).min(), df_movies['release_year'].max()
    df_movies['release_year_norm'] = (df_movies['release_year'] - min_y) / (max_y - min_y)

    # 合并电影特征表并过滤掉训练集没见过的电影
    item_features = df_movies[df_movies['movieId'].isin(movie_map)].copy()
    item_features['movieId_int'] = item_features['movieId'].map(movie_map)
    item_features = item_features.merge(df_item_stats, on='movieId', how='left').fillna(0)
    
    # 最终物品特征表：[movieId_int, genres_idx, release_year_norm, item_avg_rating_norm, item_rating_count_norm]
    item_features_final = item_features[[
        'movieId_int', 'genres_idx', 'release_year_norm', 
        'item_avg_rating_norm', 'item_rating_count_norm'
    ]].sort_values('movieId_int')

    # --- 5. 用户侧特征加工 (User Features) ---
    logger.info("Processing user features...")
    df_user_stats = calculate_user_stats(df_train)
    df_user_genres = get_user_genre_preference(df_train, df_movies, top_k=3)
    df_user_genres['user_top_genres_idx'] = df_user_genres['user_top_genres'].apply(gi.transform)

    # 合并用户特征表并过滤
    user_features = pd.DataFrame({'userId': unique_users})
    user_features['userId_int'] = user_features['userId'].map(user_map)
    user_features = user_features.merge(df_user_stats, on='userId', how='left')
    user_features = user_features.merge(df_user_genres[['userId', 'user_top_genres_idx']], on='userId', how='left')
    
    # 填充缺失的题材偏好
    user_features['user_top_genres_idx'] = user_features['user_top_genres_idx'].apply(
        lambda x: x if isinstance(x, list) else [0, 0, 0]
    )
    
    user_features_final = user_features[[
        'userId_int', 'user_avg_rating_norm', 'user_rating_count_norm', 'user_top_genres_idx'
    ]].sort_values('userId_int')

    # --- 6. 应用映射到交互集并过滤 OOV ---
    def finalize_interactions(df, name):
        mask = df['userId'].isin(user_map) & df['movieId'].isin(movie_map)
        df_out = df[mask].copy()
        df_out['userId'] = df_out['userId'].map(user_map)
        df_out['movieId'] = df_out['movieId'].map(movie_map)
        logger.info(f"{name} set: Final size {len(df_out)}")
        return df_out[['userId', 'movieId', 'rating']]

    df_train_final = finalize_interactions(df_train, "Train")
    df_val_final = finalize_interactions(df_val, "Val")
    df_test_final = finalize_interactions(df_test, "Test")

    # --- 7. 保存所有产出 ---
    logger.info("Saving all features and mappings...")
    df_train_final.to_parquet(output_dir / "train.parquet", index=False)
    df_val_final.to_parquet(output_dir / "val.parquet", index=False)
    df_test_final.to_parquet(output_dir / "test.parquet", index=False)
    
    item_features_final.to_parquet(output_dir / "item_features.parquet", index=False)
    user_features_final.to_parquet(output_dir / "user_features.parquet", index=False)
    
    with open(output_dir / "user_map.pkl", "wb") as f: pickle.dump(user_map, f)
    with open(output_dir / "movie_map.pkl", "wb") as f: pickle.dump(movie_map, f)
    with open(output_dir / "genre_indexer.pkl", "wb") as f: pickle.dump(gi, f)

    logger.success(f"Phase 1 complete! Feature tables saved in {output_dir}")

if __name__ == "__main__":
    prepare_data_v2()
