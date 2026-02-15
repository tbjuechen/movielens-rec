import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_user_chunk(user_chunk_data, movie_to_genres, genre_to_movies, all_movie_ids):
    """单个进程处理一部分用户的逻辑"""
    res_pos = []
    res_neg = []
    
    # 对传入的 chunk 再次按用户分组处理序列
    for uid, group in user_chunk_data.groupby('userId'):
        group = group.sort_values('timestamp')
        # 生成序列
        mids = group['movieId'].tolist()
        seqs = [mids[max(0, j-5):j] for j in range(len(mids))]
        
        for i, row in enumerate(group.itertuples()):
            seq = seqs[i]
            # 正样本
            res_pos.append([row.userId, row.movieId, row.timestamp, seq, 1])
            
            # 难负样本 (5个)
            m_genres = movie_to_genres.get(row.movieId, [])
            if m_genres:
                candidates = genre_to_movies.get(m_genres[0], all_movie_ids)
                hard_negs = np.random.choice(candidates, 5).tolist()
            else:
                hard_negs = np.random.choice(all_movie_ids, 5).tolist()
            
            # 随机负样本 (4个)
            rand_negs = np.random.choice(all_movie_ids, 4).tolist()
            
            for nid in (hard_negs + rand_negs):
                res_neg.append([row.userId, int(nid), row.timestamp, seq, 0])
                
    return res_pos + res_neg

def prepare_full_ranking_dataset():
    data_dir = Path("data/processed")
    output_dir = data_dir / "ranking/samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载 32M 全量交互记录...")
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    movies = pd.read_parquet(data_dir / "movies.parquet")
    
    # 建立查找表 (广播到各进程)
    movie_to_genres = dict(zip(movies['movieId'], movies['genres']))
    movies_exploded = movies.explode('genres')
    genre_to_movies = movies_exploded.groupby('genres')['movieId'].apply(lambda x: np.array(x, dtype=np.int32)).to_dict()
    all_movie_ids = movies['movieId'].unique().astype(np.int32)

    user_ids = ratings['userId'].unique()
    num_shards = 10 # 依然分 10 个文件存，防止单个文件太大
    users_per_shard = len(user_ids) // num_shards
    
    # 建议核心数：M4 性能强，建议 4-6 个进程以平衡内存
    max_workers = 6

    for shard_idx in range(num_shards):
        start_u = shard_idx * users_per_shard
        end_u = (shard_idx + 1) * users_per_shard if shard_idx < num_shards-1 else len(user_ids)
        shard_users = user_ids[start_u:end_u]
        
        df_shard_input = ratings[ratings['userId'].isin(shard_users)].copy()
        
        # 将 Shard 再次切分给多进程并行
        user_sub_chunks = np.array_split(shard_users, max_workers)
        
        logger.info(f"正在并行处理 Shard {shard_idx}...")
        shard_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for sub_users in user_sub_chunks:
                sub_df = df_shard_input[df_shard_input['userId'].isin(sub_users)]
                futures.append(executor.submit(process_user_chunk, sub_df, movie_to_genres, genre_to_movies, all_movie_ids))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Shard {shard_idx}"):
                shard_results.extend(future.result())

        # 转换并保存
        shard_df = pd.DataFrame(shard_results, columns=['userId', 'movieId', 'timestamp', 'seq_history', 'label'])
        shard_df['userId'] = shard_df['userId'].astype(np.int32)
        shard_df['movieId'] = shard_df['movieId'].astype(np.int32)
        shard_df['label'] = shard_df['label'].astype(np.int8)
        
        output_path = output_dir / f"samples_shard_{shard_idx}.parquet"
        shard_df.to_parquet(output_path, index=False)
        
        logger.success(f"Shard {shard_idx} 完成，行数: {len(shard_df):,}")
        del df_shard_input, shard_results, shard_df
        gc.collect()

if __name__ == "__main__":
    prepare_full_ranking_dataset()
