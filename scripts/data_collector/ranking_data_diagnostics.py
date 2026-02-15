import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

def run_diagnostics():
    data_dir = Path("data/processed")
    tmdb_dir = data_dir / "tmdb"
    
    logger.info("正在进行精排数据深度诊断...")

    # 1. 加载数据
    ratings = pd.read_parquet(data_dir / "ratings.parquet").sample(1000000, random_state=42)
    tmdb_movies = pd.read_parquet(tmdb_dir / "tmdb_movies.parquet")
    tmdb_crew = pd.read_parquet(tmdb_dir / "tmdb_movie_crew.parquet")
    tmdb_persons = pd.read_parquet(tmdb_dir / "tmdb_persons.parquet")

    # 2. 特征覆盖率分析
    df_merged = ratings.merge(tmdb_movies, on='movieId', how='left')
    coverage = df_merged[['overview', 'budget', 'runtime', 'vote_average']].notna().mean() * 100
    
    print("\n" + "="*60)
    print("【1. 特征覆盖率报告】")
    print("-" * 60)
    for col, rate in coverage.items():
        print(f"特征 {col:<15} | 覆盖率: {rate:>6.2f}%")

    # 3. 导演影响力深度分析
    print("\n" + "="*60)
    print("【2. 顶级导演评分拉动力 (Top 20)】")
    print("-" * 60)
    directors = tmdb_crew[tmdb_crew['job'] == 'Director'].merge(ratings, on='movieId')
    dir_stats = directors.groupby('personId').agg({'rating': ['mean', 'count', 'std']}).reset_index()
    dir_stats.columns = ['personId', 'avg_rating', 'rating_count', 'std_rating']
    
    dir_report = dir_stats[dir_stats['rating_count'] > 500].merge(tmdb_persons[['personId', 'name']], on='personId')
    dir_report = dir_report.sort_values(by='avg_rating', ascending=False).head(20)
    
    print(f"{'导演姓名':<25} | {'平均分':<10} | {'评价人数':<10} | {'稳定性(Std)':<10}")
    for _, row in dir_report.iterrows():
        name = row['name']
        print(f"{name:<25} | {row['avg_rating']:>8.2f} | {int(row['rating_count']):>8} | {row['std_rating']:>8.2f}")

    # 4. 相关性分析
    print("\n" + "="*60)
    print("【3. 特征与标签的相关性系数】")
    print("-" * 60)
    numerical_df = df_merged[['rating', 'runtime', 'budget', 'revenue', 'vote_average']].replace(0, np.nan).dropna()
    print(numerical_df.corr()['rating'].sort_values(ascending=False))

    # 5. 用户匹配忠诚度
    user_dir_repeat = directors.groupby(['userId', 'personId']).size().reset_index(name='view_count')
    loyal_interactions = user_dir_repeat[user_dir_repeat['view_count'] >= 2]
    print("\n" + "="*60)
    print("【4. 用户匹配忠诚度诊断】")
    print("-" * 60)
    print(f"『回头客』用户-导演对数: {len(loyal_interactions):,}")
    
    logger.success("诊断任务完成。")

if __name__ == "__main__":
    run_diagnostics()
