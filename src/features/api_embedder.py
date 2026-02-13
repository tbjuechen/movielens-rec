import os
from openai import OpenAI
from loguru import logger
from typing import List
import time

class APIEmbedder:
    """
    语义嵌入引擎：利用 OpenAI API 获取高质量文本向量。
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment!")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = "text-embedding-3-small"

    def get_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        批量获取嵌入向量。
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [t[:8000] for t in texts[i:i + batch_size]] # 防止单文本超长
            
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Embedding API error at batch {i}: {e}")
                # 简单指数退避
                time.sleep(5)
                return self.get_embeddings(texts[i:], batch_size) # 递归重试剩余部分
                
        return all_embeddings
