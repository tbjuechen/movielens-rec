import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from src.features.ranking_feature_engine import RankingFeatureEngine
import gc

def build_final_feature_store():
    """
    全量特征预计算：
    逐个分片读取 samples，计算交叉特征，并持久化。
    """
    data_dir = Path("data/processed")
    input_dir = data_dir / "ranking/samples"
    output_dir = data_dir / "ranking/features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 准备参考数据（用于去泄露）
    # 为保证严谨，我们加载原始 ratings，取前 80% 作为历史参考
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    ratings = ratings.sort_values('timestamp')
    history_ratings = ratings.head(int(len(ratings) * 0.8))
    
    # 2. 初始化特征引擎
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    engine.initialize(history_ratings)
    
    del ratings, history_ratings
    gc.collect()

    # 3. 逐个分片计算并存盘
    shard_files = sorted(list(input_dir.glob("samples_shard_*.parquet")))
    
    for shard_file in shard_files:
        logger.info(f"正在计算特征: {shard_file.name}...")
        df_shard = pd.read_parquet(shard_file)
        
        # 核心：批量生成特征矩阵
        feature_df = engine.build_feature_matrix(df_shard)
        
        # 仅保留模型需要的列，舍弃中间辅助列以节省空间
        cols_to_keep = [
            'userId', 'movieId', 'label', 'timestamp', 'seq_history',
            'user_avg_rating', 'user_rating_std', 'user_rating_count_log',
            'year', 'runtime', 'budget_log', 'revenue_log', 'vote_average', 'vote_count',
            'is_director_match', 'actor_match_count', 'rating_diff', 'semantic_sim', 'genre_match_score'
        ]
        feature_df = feature_df[cols_to_keep]
        
        output_path = output_dir / shard_file.name.replace("samples", "features")
        feature_df.to_parquet(output_path, index=False)
        
        logger.success(f"特征分片已存至 {output_path}")
        
        del df_shard, feature_df
        gc.collect()

    logger.success("全量特征工程持久化完成！")

if __name__ == "__main__":
    build_final_feature_store()
