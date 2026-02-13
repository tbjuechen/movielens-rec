import os
from openai import OpenAI
from loguru import logger
from typing import List
import time

class APIEmbedder:
    """
    语义嵌入引擎：利用 API 获取高质量文本向量。
    支持 OpenAI, BGE-M3 等兼容接口。
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        # 允许从环境变量读取模型名称，默认为 OpenAI 的模型
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            logger.error("API Key not found in environment!")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"APIEmbedder initialized with model: {self.model}")

    def get_embeddings(self, texts: List[str], batch_size: int = 500) -> List[List[float]]:
        """
        批量获取嵌入向量。由于 API 限制较高 (RPM 2000, TPM 100W)，增加 batch_size 以加速。
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t)[:8000] for t in texts[i:i + batch_size]]
            
            try:
                # 记录请求时间
                start_time = time.time()
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model
                )
                duration = time.time() - start_time
                logger.debug(f"Fetched {len(batch_texts)} embeddings in {duration:.2f}s")
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Embedding API error at batch {i}: {e}")
                # 简单指数退避
                time.sleep(5)
                return self.get_embeddings(texts[i:], batch_size) # 递归重试剩余部分
                
        return all_embeddings
