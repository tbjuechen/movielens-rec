import pandas as pd
from pathlib import Path
from loguru import logger
import pickle

def prepare_data():
    input_path = Path("data/processed/ratings.parquet")
    output_dir = Path("data/processed/two_tower")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info("Loading ratings data...")
    df = pd.read_parquet(input_path)
    
    # 1. 按时间排序
    logger.info("Sorting by timestamp...")
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # 2. 计算切分点
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    logger.info(f"Initial split - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # 3. ID 映射 (仅基于训练集建立映射)
    logger.info("Generating ID mappings based on training set...")
    
    unique_users = df_train['userId'].unique()
    unique_movies = df_train['movieId'].unique()
    
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    movie_map = {mid: i for i, mid in enumerate(unique_movies)}
    
    # 保存映射表以便后续使用
    with open(output_dir / "user_map.pkl", "wb") as f:
        pickle.dump(user_map, f)
    with open(output_dir / "movie_map.pkl", "wb") as f:
        pickle.dump(movie_map, f)
        
    logger.info(f"Vocab size - Users: {len(user_map)}, Movies: {len(movie_map)}")

    # 4. 应用映射并过滤 OOV (Out-of-Vocabulary)
    def process_split(df_split, name):
        initial_count = len(df_split)
        # 过滤掉训练集中没见过的用户和电影
        mask = df_split['userId'].isin(user_map) & df_split['movieId'].isin(movie_map)
        df_split = df_split[mask].copy()
        
        # 转换 ID
        df_split['userId'] = df_split['userId'].map(user_map)
        df_split['movieId'] = df_split['movieId'].map(movie_map)
        
        logger.info(f"{name} set: filtered {initial_count - len(df_split)} OOV records. Final size: {len(df_split)}")
        return df_split

    df_train['userId'] = df_train['userId'].map(user_map)
    df_train['movieId'] = df_train['movieId'].map(movie_map)
    
    df_val = process_split(df_val, "Validation")
    df_test = process_split(df_test, "Test")

    # 5. 保存结果
    logger.info("Saving processed datasets...")
    df_train.to_parquet(output_dir / "train.parquet", index=False)
    df_val.to_parquet(output_dir / "val.parquet", index=False)
    df_test.to_parquet(output_dir / "test.parquet", index=False)
    
    logger.success(f"Two-tower data preparation complete! Files saved in {output_dir}")

if __name__ == "__main__":
    prepare_data()
