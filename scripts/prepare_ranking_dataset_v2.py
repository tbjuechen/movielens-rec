import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import gc

def prepare_full_ranking_dataset():
    """
    全量样本构造 (32M -> 3.2亿行预备):
    1. 内存优化：分片处理 + 类型压缩
    2. 全量覆盖
    3. 1:9 混合负采样
    """
    data_dir = Path("data/processed")
    output_dir = data_dir / "ranking/samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    logger.info("加载 32M 全量交互记录...")
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    # 压缩类型以省内存
    ratings['userId'] = ratings['userId'].astype(np.int32)
    ratings['movieId'] = ratings['movieId'].astype(np.int32)
    ratings['label'] = 1
    
    movies = pd.read_parquet(data_dir / "movies.parquet")
    all_movie_ids = movies['movieId'].unique().astype(np.int32)
    
    # 建立题材映射用于 Hard Negative
    movies_exploded = movies.explode('genres')
    genre_to_movies = movies_exploded.groupby('genres')['movieId'].apply(lambda x: np.array(x, dtype=np.int32)).to_dict()
    movie_to_genres = dict(zip(movies['movieId'], movies['genres']))

    # 2. 按用户分组，准备分片处理
    logger.info("按用户分组并开始分片处理...")
    user_ids = ratings['userId'].unique()
    # 每 20,000 个用户一个分片 (约 300w 条交互)
    chunk_size = 20000 
    
    for shard_idx, i in enumerate(range(0, len(user_ids), chunk_size)):
        batch_users = user_ids[i:i+chunk_size]
        df_shard = ratings[ratings['userId'].isin(batch_users)].sort_values(['userId', 'timestamp']).copy()
        
        # A. 生成序列特征
        df_shard['seq_history'] = df_shard.groupby('userId')['movieId'].transform(
            lambda x: [x.iloc[max(0, j-5):j].tolist() for j in range(len(x))]
        )
        
        # B. 1:9 混合负采样
        logger.info(f"正在处理 Shard {shard_idx} (用户数: {len(batch_users)})...")
        
        all_pos_data = []
        all_neg_data = []
        
        for row in tqdm(df_shard.itertuples(), total=len(df_shard), leave=False):
            # 正样本
            all_pos_data.append([row.userId, row.movieId, row.timestamp, row.seq_history, 1])
            
            # 难负样本 (5个)
            m_genres = movie_to_genres.get(row.movieId, [])
            if isinstance(m_genres, (list, np.ndarray)) and len(m_genres) > 0:
                candidates = genre_to_movies.get(m_genres[0], all_movie_ids)
                hard_negs = np.random.choice(candidates, 5)
            else:
                hard_negs = np.random.choice(all_movie_ids, 5)
            
            # 随机负样本 (4个)
            rand_negs = np.random.choice(all_movie_ids, 4)
            
            for nid in np.concatenate([hard_negs, rand_negs]):
                all_neg_data.append([row.userId, int(nid), row.timestamp, row.seq_history, 0])

        # C. 转化为 DataFrame 并保存
        shard_df = pd.DataFrame(all_pos_data + all_neg_data, 
                                columns=['userId', 'movieId', 'timestamp', 'seq_history', 'label'])
        
        # 极致压缩保存
        shard_df['userId'] = shard_df['userId'].astype(np.int32)
        shard_df['movieId'] = shard_df['movieId'].astype(np.int32)
        shard_df['label'] = shard_df['label'].astype(np.int8)
        
        output_path = output_dir / f"samples_shard_{shard_idx}.parquet"
        shard_df.to_parquet(output_path, index=False, compression='snappy')
        
        logger.success(f"Shard {shard_idx} 已保存至 {output_path} | 行数: {len(shard_df):,}")
        
        # 强制垃圾回收
        del df_shard, shard_df, all_pos_data, all_neg_data
        gc.collect()

    logger.success("全量样本构造完成！")

if __name__ == "__main__":
    prepare_full_ranking_dataset()
