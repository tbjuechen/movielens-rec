import os
from openai import OpenAI
from loguru import logger
from typing import List
import time
from concurrent.futures import ThreadPoolExecutor

class APIEmbedder:
    """
    语义嵌入引擎：利用 API 获取高质量文本向量。
    支持多线程并发以利用高 RPM 限额。
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            logger.error("API Key not found in environment!")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"APIEmbedder initialized with model: {self.model}")

    def _fetch_single_batch(self, batch_texts: List[str]) -> List[List[float]]:
        """单个 Batch 的抓取逻辑，用于线程池"""
        try:
            response = self.client.embeddings.create(
                input=batch_texts,
                model=self.model
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Batch fetch error: {e}")
            time.sleep(2) # 遇到错误稍作休息
            return self._fetch_single_batch(batch_texts) # 简单重试

    def get_embeddings(self, texts: List[str], batch_size: int = 64, max_workers: int = 10) -> List[List[float]]:
        """
        利用线程池并发获取嵌入向量。
        """
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_embeddings = []
        
        logger.info(f"Starting parallel embedding fetch with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用 map 保证返回顺序与输入顺序一致
            results = list(executor.map(self._fetch_single_batch, batches))
            
        for batch_res in results:
            all_embeddings.extend(batch_res)
                
        return all_embeddings
