import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.data_collector.tmdb_client import TMDBClient
from loguru import logger
from tqdm import tqdm

def save_json(data, path):
    """
    保存全量 Raw JSON 原始响应。
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def crawl_task(tmdb_id, client, cache_dir):
    """
    单条抓取任务单元。
    """
    try:
        # 转换 ID 为整数避免文件名格式不统一
        tid = int(tmdb_id)
        file_path = cache_dir / f"{tid}.json"
        
        # 1. 断点续传逻辑
        if file_path.exists():
            return "skipped"
        
        # 2. 抓取
        data = client.get_full_movie_info(tid)
        if data:
            save_json(data, file_path)
            return "success"
        return "failed"
    except Exception as e:
        return "failed"

def run_batch_crawl():
    # 大幅提升线程数 (M4 性能强劲，可以处理更高的网络并发)
    MAX_WORKERS = 50 
    
    # 1. 加载待抓取列表
    links_path = Path("data/processed/links.parquet")
    if not links_path.exists():
        logger.error("请先运行基础预处理脚本以生成 links.parquet")
        return
    
    links = pd.read_parquet(links_path)
    # 获取唯一的有效 TMDb ID
    target_ids = links['tmdbId'].dropna().unique().tolist()
    
    client = TMDBClient()
    if not client.api_key:
        return

    cache_dir = Path("data/raw/tmdb_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"启动批量抓取任务。目标总数: {len(target_ids)}")
    
    results_stats = {"success": 0, "failed": 0, "skipped": 0}
    
    # 2. 执行并发抓取
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(crawl_task, tid, client, cache_dir): tid for tid in target_ids}
        
        with tqdm(as_completed(futures), total=len(target_ids), desc="TMDb 抓取中", mininterval=1.0) as pbar:
            for future in pbar:
                res = future.result()
                results_stats[res] += 1
                # 实时更新统计信息
                if results_stats["success"] % 100 == 0:
                    pbar.set_postfix(results_stats)
            
    logger.success(f"任务圆满结束！汇总统计: {results_stats}")

if __name__ == "__main__":
    run_batch_crawl()
