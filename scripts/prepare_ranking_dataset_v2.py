import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm

def prepare_ranking_dataset_v2():
    """
    精排样本构造 Pro 版：
    1. 1:9 采样比例
    2. 引入 Hard Negatives (同题材负采样)
    3. 扩大数据量至 500w+ 样本
    """
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    ranking_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    logger.info("正在加载 MovieLens 数据...")
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    movies = pd.read_parquet(data_dir / "movies.parquet")
    
    # 增加处理量：取前 200 万条原始交互作为正样本源
    df = ratings.sort_values(['userId', 'timestamp']).head(2000000).copy()
    
    # 2. 生成序列特征
    logger.info("生成历史行为序列...")
    df['seq_history'] = df.groupby('userId')['movieId'].transform(
        lambda x: [x.iloc[max(0, i-5):i].tolist() for i in range(len(x))]
    )

    # 3. 准备负采样资源
    all_movie_ids = movies['movieId'].unique()
    # 建立题材 -> 电影列表的映射，用于 Hard Negative
    movies_exploded = movies.explode('genres')
    genre_to_movies = movies_exploded.groupby('genres')['movieId'].apply(list).to_dict()

    logger.info("执行 1:9 混合负采样 (4 随机 + 5 难负样本)...")
    
    def get_mixed_negatives(row):
        # 随机负样本 (4个)
        rand_negs = np.random.choice(all_movie_ids, 4).tolist()
        
        # 难负样本 (5个)：从该电影的第一个题材中抽取
        target_genres = row['genres']
        hard_negs = []
        if isinstance(target_genres, (list, np.ndarray)) and len(target_genres) > 0:
            primary_genre = target_genres[0]
            candidates = genre_to_movies.get(primary_genre, all_movie_ids)
            hard_negs = np.random.choice(candidates, 5).tolist()
        else:
            hard_negs = np.random.choice(all_movie_ids, 5).tolist()
            
        return rand_negs + hard_negs

    # 联表获取题材信息
    df = df.merge(movies[['movieId', 'genres']], on='movieId', how='left')
    
    # 构造负样本组
    neg_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sampling"):
        negs = get_mixed_negatives(row)
        for i, nid in enumerate(negs):
            neg_list.append({
                'userId': row['userId'],
                'movieId': nid,
                'timestamp': row['timestamp'],
                'seq_history': row['seq_history'],
                'label': 0
            })
            
    df['label'] = 1
    # 合并
    final_samples = pd.concat([df.drop(columns=['genres']), pd.DataFrame(neg_list)], ignore_index=True)
    
    output_path = ranking_dir / "ranking_samples_v2_hard.parquet"
    final_samples.to_parquet(output_path, index=False)
    
    logger.success(f"Pro 版样本集生成完成！总行数: {len(final_samples):,}")
    logger.info(f"保存路径: {output_path}")

if __name__ == "__main__":
    prepare_ranking_dataset_v2()
