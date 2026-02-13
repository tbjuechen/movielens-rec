import requests
from bs4 import BeautifulSoup
from loguru import logger
import time

class IMDbSpider:
    """
    IMDb 网页爬虫类
    """
    def __init__(self):
        self.base_url = "https://www.imdb.com/title/tt"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }

    def fetch_movie_info(self, imdb_id: str) -> dict:
        """
        获取单部电影的剧情简介。
        """
        # IMDb ID 格式化为 7 位或 8 位
        formatted_id = str(imdb_id).zfill(7)
        url = f"{self.base_url}{formatted_id}/"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 获取剧情简介 (从 Open Graph 标签)
            desc_tag = soup.find("meta", property="og:description")
            summary = desc_tag["content"] if desc_tag else ""
            
            return {
                "imdbId": imdb_id,
                "summary": summary,
                "url": url
            }
            
        except Exception as e:
            logger.warning(f"Failed to fetch {imdb_id}: {e}")
            return None
