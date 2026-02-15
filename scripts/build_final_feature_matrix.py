import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from src.features.ranking_feature_engine import RankingFeatureEngine

def build_final_store():
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    input_path = ranking_dir / "ranking_samples_v2_hard.parquet"
    output_path = ranking_dir / "final_ranking_feature_matrix.parquet"

    if not input_path.exists():
        logger.error("V2 样本集不存在，请先运行 scripts/prepare_ranking_dataset_v2.py")
        return

    logger.info("正在加载 2000w+ 样本数据进行特征预计算...")
    samples_df = pd.read_parquet(input_path)
    
    # 执行三段式隔离
    n = len(samples_df)
    # 此处假设 60% 依然是历史参考线
    history_ratings = samples_df.iloc[:int(n*0.6)]
    history_ratings = history_ratings[history_ratings['label'] == 1][['userId', 'movieId', 'timestamp']]

    # 初始化引擎
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    engine.initialize(history_ratings)
    
    # 核心：计算全量特征
    # 针对大规模数据，我们分块处理以防止内存溢出
    logger.info("开始分块提取特征并持久化...")
    chunk_size = 500000
    all_chunks = []
    
    for i in range(0, len(samples_df), chunk_size):
        chunk = samples_df.iloc[i:i+chunk_size]
        logger.info(f"Processing chunk {i//chunk_size + 1}...")
        feature_chunk = engine.build_feature_matrix(chunk)
        all_chunks.append(feature_chunk)
        
    final_df = pd.concat(all_chunks, ignore_index=True)
    
    # 保存大宽表
    logger.info(f"保存大宽表至 {output_path}...")
    final_df.to_parquet(output_path, index=False)
    logger.success("特征工程持久化完成！")

if __name__ == "__main__":
    build_final_store()
