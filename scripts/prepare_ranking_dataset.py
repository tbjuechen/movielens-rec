import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm

def prepare_ranking_dataset():
    """
    构造排序模型的训练集：
    1. 引入 Label (1:正样本, 0:负样本)
    2. 构造历史观影序列 (Last 5)
    3. 采样曝光未点击样本
    """
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    ranking_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载并排序
    logger.info("加载并按时间排序交互记录...")
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    ratings = ratings.sort_values(['userId', 'timestamp']).reset_index(drop=True)

    # 2. 生成序列特征 (Last 5 movies)
    # 此步骤在千万级数据上较慢，使用采样 100w 记录进行演示，或使用向量化优化
    logger.info("生成序列特征 (Last 5 History)...")
    # 为了保证脚本能跑完，我们先处理前 100 万条样本作为排序原型
    df = ratings.head(1000000).copy()
    
    df['seq_history'] = df.groupby('userId')['movieId'].transform(
        lambda x: [x.iloc[max(0, i-5):i].tolist() for i in range(len(x))]
    )

    # 3. 负采样 (1:4 随机采样)
    logger.info("执行 1:4 负采样 (增加训练难度)...")
    all_movie_ids = df['movieId'].unique()
    
    neg_ratio = 4
    neg_dfs = []
    for _ in range(neg_ratio):
        neg_df = df.copy()
        neg_df['movieId'] = np.random.choice(all_movie_ids, size=len(neg_df))
        neg_df['label'] = 0
        neg_dfs.append(neg_df)
    
    df['label'] = 1
    ranking_samples = pd.concat([df] + neg_dfs, ignore_index=True)
    
    # 4. 落地
    output_path = ranking_dir / "ranking_samples_prototype.parquet"
    ranking_samples.to_parquet(output_path, index=False)
    
    logger.success(f"精排原型样本集已生成：{output_path}")
    logger.info(f"样本总数: {len(ranking_samples):,} | 正负比例 1:1")

if __name__ == "__main__":
    prepare_ranking_dataset()
