from scripts.data_collector.tmdb_client import TMDBClient
from loguru import logger
import json

def test_connection():
    client = TMDBClient()
    # 862 是 Toy Story 的 TMDb ID
    test_id = 862
    
    logger.info(f"正在尝试抓取电影 ID {test_id} 的全量特征...")
    data = client.get_full_movie_info(test_id)
    
    if data:
        logger.success("连接成功！抓取数据预览：")
        print("\n" + "="*60)
        print(f"标题: {data.get('title')}")
        print(f"简介: {data.get('overview')[:100]}...")
        print(f"上映日期: {data.get('release_date')}")
        print(f"时长: {data.get('runtime')} 分钟")
        
        # 验证关键词
        keywords = [k['name'] for k in data.get('keywords', {}).get('keywords', [])]
        print(f"关键词 (前5个): {keywords[:5]}")
        
        # 验证演职员
        director = [m['name'] for m in data.get('credits', {}).get('crew', []) if m['job'] == 'Director']
        cast = [m['name'] for m in data.get('credits', {}).get('cast', [])]
        print(f"导演: {director}")
        print(f"主演 (前3名): {cast[:3]}")
        print("="*60 + "\n")
    else:
        logger.error("抓取失败，请检查 .env 中的 API Key 是否正确。")

if __name__ == "__main__":
    test_connection()
