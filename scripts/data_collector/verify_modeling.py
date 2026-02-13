import pandas as pd
from pathlib import Path
from loguru import logger

def verify_modeling():
    data_dir = Path("data/processed/tmdb")
    tables = ["tmdb_movies", "tmdb_persons", "tmdb_movie_cast", "tmdb_movie_crew"]
    
    logger.info("开始数据建模质量检查...")
    
    dfs = {}
    for t in tables:
        path = data_dir / f"{t}.parquet"
        if not path.exists():
            logger.error(f"缺失关键表: {t}")
            return
        dfs[t] = pd.read_parquet(path)
        logger.info(f"表 {t:<16} | 行数: {len(dfs[t]):,}")

    # 1. 关联完整性检查 (Referential Integrity)
    movie_ids = set(dfs['tmdb_movies']['movieId'])
    cast_movie_ids = set(dfs['tmdb_movie_cast']['movieId'])
    missing_movies_in_cast = cast_movie_ids - movie_ids
    if missing_movies_in_cast:
        logger.warning(f"警告：演员表中有 {len(missing_movies_in_cast)} 部电影在主表中未找到！")
    else:
        logger.success("关联检查通过：所有关系表中的电影均存在于主表中。")

    # 2. 人员去重检查
    person_ids = dfs['tmdb_persons']['personId']
    if person_ids.nunique() == len(person_ids):
        logger.success("去重检查通过：人员表中的 ID 是唯一的。")
    else:
        logger.error("去重检查失败：人员表中存在重复 ID！")

    # 3. 核心特征缺失普查
    logger.info("核心特征缺失率普查:")
    cols_to_check = {
        'tmdb_movies': ['overview', 'budget', 'runtime'],
        'tmdb_persons': ['name']
    }
    for table, cols in cols_to_check.items():
        for col in cols:
            missing_rate = dfs[table][col].isna().mean() * 100
            # 特殊处理 budget=0 的情况（TMDb 很多默认填0）
            if col == 'budget':
                zero_rate = (dfs[table][col] == 0).mean() * 100
                logger.info(f"- {table}.{col:<10} | 缺失率: {missing_rate:.2f}% | 零值率: {zero_rate:.2f}%")
            else:
                logger.info(f"- {table}.{col:<10} | 缺失率: {missing_rate:.2f}%")

    # 4. 统计分析预览
    logger.info("快速统计预览:")
    avg_cast = len(dfs['tmdb_movie_cast']) / len(dfs['tmdb_movies'])
    logger.info(f"- 平均每部电影包含演员: {avg_cast:.1f} 名")
    
    logger.success("数据体检结束。")

if __name__ == "__main__":
    verify_modeling()
