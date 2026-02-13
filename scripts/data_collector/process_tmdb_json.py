import pandas as pd
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm

def process_tmdb_jsons():
    cache_dir = Path("data/raw/tmdb_cache")
    output_path = Path("data/processed/tmdb_metadata.parquet")
    links_path = Path("data/processed/links.parquet")

    if not cache_dir.exists():
        logger.error(f"缓存目录 {cache_dir} 不存在！")
        return

    # 1. 加载映射表以便对齐 movieId
    links = pd.read_parquet(links_path)
    # 建立 tmdbId -> movieId 的映射
    tmdb_to_movie = dict(zip(links['tmdbId'].dropna().astype(int), links['movieId']))

    json_files = list(cache_dir.glob("*.json"))
    logger.info(f"开始处理 {len(json_files)} 个 JSON 文件...")

    processed_data = []

    for file_path in tqdm(json_files, desc="处理 JSON"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取核心字段
            tmdb_id = int(data.get('id', 0))
            if tmdb_id not in tmdb_to_movie:
                continue
                
            # 提取导演
            crew = data.get('credits', {}).get('crew', [])
            directors = [m['name'] for m in crew if m['job'] == 'Director']
            
            # 提取前 5 名演员
            cast = [m['name'] for m in data.get('credits', {}).get('cast', [])[:5]]
            
            # 提取关键词
            keywords = [k['name'] for k in data.get('keywords', {}).get('keywords', [])]
            
            processed_data.append({
                'movieId': tmdb_to_movie[tmdb_id],
                'tmdbId': tmdb_id,
                'overview': data.get('overview', ''),
                'runtime': data.get('runtime', 0),
                'vote_average': data.get('vote_average', 0.0),
                'vote_count': data.get('vote_count', 0),
                'release_date': data.get('release_date', ''),
                'directors': directors,
                'cast': cast,
                'keywords': keywords,
                'poster_path': data.get('poster_path', '')
            })
        except Exception as e:
            logger.warning(f"解析文件 {file_path} 出错: {e}")

    # 2. 转换为 DataFrame 并保存
    if processed_data:
        df_tmdb = pd.DataFrame(processed_data)
        df_tmdb.to_parquet(output_path, index=False)
        logger.success(f"处理完成！整合后的数据已保存至 {output_path}")
        logger.info(f"最终表规模: {df_tmdb.shape}")
    else:
        logger.warning("未处理任何有效数据。")

if __name__ == "__main__":
    process_tmdb_jsons()
