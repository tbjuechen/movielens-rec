import pandas as pd
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm

def process_tmdb_jsons_normalized():
    cache_dir = Path("data/raw/tmdb_cache")
    output_dir = Path("data/processed/tmdb")
    output_dir.mkdir(parents=True, exist_ok=True)
    links_path = Path("data/processed/links.parquet")

    if not cache_dir.exists():
        logger.error(f"缓存目录 {cache_dir} 不存在！")
        return

    # 1. 加载映射表以便对齐 movieId
    links = pd.read_parquet(links_path)
    tmdb_to_movie = dict(zip(links['tmdbId'].dropna().astype(int), links['movieId']))

    json_files = list(cache_dir.glob("*.json"))
    logger.info(f"开始处理 {len(json_files)} 个 JSON 文件并执行星型建模...")

    # 数据容器
    movies_list = []
    persons_dict = {} # 使用字典确保 personId 唯一
    cast_links = []
    crew_links = []

    for file_path in tqdm(json_files, desc="解析 JSON"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tmdb_id = int(data.get('id', 0))
            if tmdb_id not in tmdb_to_movie:
                continue
            
            movie_id = tmdb_to_movie[tmdb_id]

            # --- A. 提取电影主信息 ---
            movies_list.append({
                'movieId': movie_id,
                'tmdbId': tmdb_id,
                'overview': data.get('overview', ''),
                'runtime': data.get('runtime', 0),
                'budget': data.get('budget', 0),
                'revenue': data.get('revenue', 0),
                'release_date': data.get('release_date', ''),
                'vote_average': data.get('vote_average', 0.0),
                'vote_count': data.get('vote_count', 0),
                'poster_path': data.get('poster_path', ''),
                'original_language': data.get('original_language', 'en')
            })

            # --- B. 提取演员与关系 ---
            cast_data = data.get('credits', {}).get('cast', [])
            for member in cast_data:
                pid = member.get('id')
                if pid:
                    # 存入唯一人员表
                    if pid not in persons_dict:
                        persons_dict[pid] = {
                            'personId': pid,
                            'name': member.get('name', ''),
                            'gender': member.get('gender', 0),
                            'profile_path': member.get('profile_path', '')
                        }
                    # 存入关系表
                    cast_links.append({
                        'movieId': movie_id,
                        'personId': pid,
                        'character': member.get('character', ''),
                        'order': member.get('order', 999) # order 越小代表戏份越重
                    })

            # --- C. 提取团队(导演等)与关系 ---
            crew_data = data.get('credits', {}).get('crew', [])
            for member in crew_data:
                pid = member.get('id')
                if pid:
                    if pid not in persons_dict:
                        persons_dict[pid] = {
                            'personId': pid,
                            'name': member.get('name', ''),
                            'gender': member.get('gender', 0),
                            'profile_path': member.get('profile_path', '')
                        }
                    crew_links.append({
                        'movieId': movie_id,
                        'personId': pid,
                        'job': member.get('job', ''),
                        'department': member.get('department', '')
                    })

        except Exception as e:
            logger.warning(f"解析文件 {file_path} 出错: {e}")

    # 2. 转换并保存四张表
    if movies_list:
        logger.info("正在保存 Parquet 和 CSV 文件...")
        
        # 定义保存逻辑
        def save_table(df_list, table_name):
            df = pd.DataFrame(df_list)
            # 保存 Parquet (生产用)
            df.to_parquet(output_dir / f"{table_name}.parquet", index=False)
            # 保存 CSV (分享用)
            df.to_csv(output_dir / f"{table_name}.csv", index=False)
            return len(df)

        n_movies = save_table(movies_list, "tmdb_movies")
        n_persons = save_table(list(persons_dict.values()), "tmdb_persons")
        n_cast = save_table(cast_links, "tmdb_movie_cast")
        n_crew = save_table(crew_links, "tmdb_movie_crew")

        logger.success(f"数据建模处理完成！文件已保存至 {output_dir}")
        logger.info(f"统计：电影 {n_movies} 部, 人员 {n_persons} 名, 演员关系 {n_cast} 条, 团队关系 {n_crew} 条")
    else:
        logger.warning("未处理任何有效数据。")

if __name__ == "__main__":
    process_tmdb_jsons_normalized()
