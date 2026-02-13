import pandas as pd
from scripts.spider.imdb_spider import IMDbSpider
from loguru import logger
import time
from pathlib import Path

def run_test_crawl():
    # 1. 确保数据存在
    links_path = Path("data/processed/links.parquet")
    if not links_path.exists():
        logger.error("links.parquet not found!")
        return
    
    links = pd.read_parquet(links_path)
    # 取前 5 个作为 Demo
    test_ids = links['imdbId'].head(5).tolist()
    
    spider = IMDbSpider()
    results = []
    
    logger.info(f"Starting test crawl for {len(test_ids)} movies...")
    
    for imdb_id in test_ids:
        info = spider.fetch_movie_info(imdb_id)
        if info:
            results.append(info)
            logger.success(f"Fetched info for IMDb ID: {imdb_id}")
        
        # 强制休眠 1 秒，保持礼貌
        time.sleep(1)
    
    # 2. 打印结果
    df_res = pd.DataFrame(results)
    print("
" + "="*60)
    print(f"{'IMDb ID':<10} | {'Movie Summary (First 100 chars)':<50}")
    print("-" * 60)
    for _, row in df_res.iterrows():
        summary = (row['summary'][:47] + '...') if len(row['summary']) > 50 else row['summary']
        print(f"{row['imdbId']:<10} | {summary:<50}")
    print("="*60 + "
")

if __name__ == "__main__":
    run_test_crawl()
