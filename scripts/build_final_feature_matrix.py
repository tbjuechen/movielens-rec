import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from src.features.ranking_feature_engine import RankingFeatureEngine
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

def process_single_shard(shard_file, history_ratings, data_dir, output_dir):
    """单个分片的特征提取进程"""
    # 重新初始化引擎（每个进程需要自己的引擎实例）
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    engine.initialize(history_ratings)
    
    df_shard = pd.read_parquet(shard_file)
    feature_df = engine.build_feature_matrix(df_shard)
    
    cols_to_keep = [
        'userId', 'movieId', 'label', 'timestamp', 'seq_history',
        'user_avg_rating', 'user_rating_std', 'user_rating_count_log',
        'year', 'runtime', 'budget_log', 'revenue_log', 'vote_average', 'vote_count',
        'is_director_match', 'actor_match_count', 'rating_diff', 'semantic_sim', 'genre_match_score'
    ]
    feature_df = feature_df[cols_to_keep]
    
    output_path = output_dir / shard_file.name.replace("samples", "features")
    feature_df.to_parquet(output_path, index=False)
    return output_path.name

def build_final_feature_store():
    data_dir = Path("data/processed")
    input_dir = data_dir / "ranking/samples"
    output_dir = data_dir / "ranking/features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 准备参考数据（共享内存）
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    ratings = ratings.sort_values('timestamp')
    history_ratings = ratings.head(int(len(ratings) * 0.8))
    del ratings
    gc.collect()

    shard_files = sorted(list(input_dir.glob("samples_shard_*.parquet")))
    
    # 限制并发数：特征提取涉及大量向量运算，建议同时跑 2-3 个
    max_workers = 3
    
    logger.info(f"开启多进程特征提取，并发数: {max_workers}...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_shard, f, history_ratings, data_dir, output_dir) for f in shard_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Feature Matrix Pro"):
            saved_name = future.result()
            logger.success(f"已完成: {saved_name}")

    logger.success("全量特征工程持久化并行处理完成！")

if __name__ == "__main__":
    build_final_feature_store()
