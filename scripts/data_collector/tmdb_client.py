import os
import requests
import time
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class TMDBClient:
    """
    TMDb API 客户端：负责全量抓取电影元数据。
    """
    def __init__(self):
        self.api_key = os.getenv("TMDB_API_KEY")
        self.base_url = "https://api.themoviedb.org/3"
        if not self.api_key:
            logger.error("环境变量 TMDB_API_KEY 缺失！请在 .env 文件中配置。")

    def get_full_movie_info(self, tmdb_id: int) -> dict:
        """
        利用 append_to_response 一次性抓取详情、演职员表和关键词。
        """
        if not self.api_key:
            return None

        url = f"{self.base_url}/movie/{int(tmdb_id)}"
        params = {
            "api_key": self.api_key,
            "append_to_response": "credits,keywords",
            "language": "en-US" # 推荐系统通常使用英文语料进行 Embedding
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            
            # 处理频率限制 (Rate Limiting)
            if response.status_code == 429:
                logger.warning("触发频率限制，休眠后重试...")
                time.sleep(2)
                return self.get_full_movie_info(tmdb_id)
            
            if response.status_code == 404:
                return None
                
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"抓取 TMDb ID {tmdb_id} 失败: {e}")
            return None
