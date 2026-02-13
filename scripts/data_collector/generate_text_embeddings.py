import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv
from src.features.api_embedder import APIEmbedder

load_dotenv()

def run_embedding_task():
    input_path = Path("data/processed/tmdb/tmdb_movies.parquet")
    output_path = Path("data/processed/tmdb/tmdb_embeddings_full.parquet")
    
    if not input_path.exists():
        logger.error("Required TMDb data not found!")
        return

    # 1. 加载待处理数据
    df = pd.read_parquet(input_path)
    # 过滤掉空简介，只处理有意义的文本
    df = df[df['overview'].str.strip().str.len() > 0].copy()
    
    logger.info(f"Total movies with overviews: {len(df)}")

    # 2. 断点续传检查
    if output_path.exists():
        df_existing = pd.read_parquet(output_path)
        existing_ids = set(df_existing['movieId'])
        df_todo = df[~df['movieId'].isin(existing_ids)].copy()
        logger.info(f"Resuming task. Found {len(existing_ids)} existing, to-do: {len(df_todo)}")
    else:
        df_todo = df
        df_existing = pd.DataFrame(columns=['movieId', 'embedding'])
        logger.info(f"Starting new task. To-do: {len(df_todo)}")

    if df_todo.empty:
        logger.success("All overviews already embedded!")
        return

    # 3. 开始批量提取
    embedder = APIEmbedder()
    chunk_size = 500
    
    with tqdm(total=len(df_todo), desc="Embedding Progress") as pbar:
        for i in range(0, len(df_todo), chunk_size):
            chunk = df_todo.iloc[i : i + chunk_size]
            
            embeddings = embedder.get_embeddings(chunk['overview'].tolist())
            
            df_chunk_res = pd.DataFrame({
                'movieId': chunk['movieId'].values,
                'embedding': embeddings
            })
            
            # 追加并保存
            df_existing = pd.concat([df_existing, df_chunk_res], ignore_index=True)
            # 保存为 Parquet (生产用)
            df_existing.to_parquet(output_path, index=False)
            # 保存为 CSV (分享与查看用)
            df_existing.to_csv(output_path.with_suffix('.csv'), index=False)
            
            pbar.update(len(chunk))
            pbar.set_postfix({"total": len(df_existing)})
    
    logger.success(f"All embeddings saved to {output_path}")

if __name__ == "__main__":
    run_embedding_task()
