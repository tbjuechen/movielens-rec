import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import faiss
from typing import List, Tuple, Dict
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from src.models.recall.base import BaseRecall

# --- PyTorch 模型定义 ---

class UserTower(nn.Module):
    def __init__(self, num_users: int, embed_dim: int):
        super(UserTower, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, user_ids):
        out = self.user_embed(user_ids)
        out = self.mlp(out)
        return F.normalize(out, p=2, dim=1) # L2 归一化

class ItemTower(nn.Module):
    def __init__(self, num_items: int, embed_dim: int):
        super(ItemTower, self).__init__()
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, item_ids):
        out = self.item_embed(item_ids)
        out = self.mlp(out)
        return F.normalize(out, p=2, dim=1) # L2 归一化

class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 64, temperature: float = 0.07):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(num_users, embed_dim)
        self.item_tower = ItemTower(num_items, embed_dim)
        self.temperature = temperature

    def forward(self, user_ids, item_ids):
        user_vector = self.user_tower(user_ids)
        item_vector = self.item_tower(item_ids)
        return user_vector, item_vector

# --- 数据集类 ---

class MovieLensDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.users = torch.LongTensor(df['userId'].values)
        self.items = torch.LongTensor(df['movieId'].values)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

# --- 召回封装类 ---

class TwoTowerRecall(BaseRecall):
    def __init__(self, embed_dim: int = 64, temperature: float = 0.07):
        super().__init__(name="TwoTowerRecall")
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.model = None
        self.faiss_index = None
        self.id_to_movie = {} # 训练集内的 index 到原始 movieId 的映射

    def train(self, df_train: pd.DataFrame, **kwargs) -> None:
        """
        PyTorch 训练循环：使用 In-batch 负采样。
        """
        num_users = df_train['userId'].max() + 1
        num_items = df_train['movieId'].max() + 1
        epochs = kwargs.get('epochs', 5)
        batch_size = kwargs.get('batch_size', 1024)
        lr = kwargs.get('lr', 0.001)
        
        # 支持 CUDA (NVIDIA), MPS (Apple Silicon), CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        logger.info(f"Training Two-Tower on {device} (Users: {num_users}, Items: {num_items})")
        
        self.model = TwoTowerModel(num_users, num_items, self.embed_dim, self.temperature).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        dataset = MovieLensDataset(df_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for user_ids, item_ids in loader:
                user_ids, item_ids = user_ids.to(device), item_ids.to(device)
                
                user_vec, item_vec = self.model(user_ids, item_ids)
                
                # 计算内积矩阵 [Batch, Batch]
                logits = torch.matmul(user_vec, item_vec.T) / self.temperature
                
                # 对角线是正样本，其余是负样本
                labels = torch.arange(user_vec.size(0)).long().to(device)
                loss = F.cross_entropy(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

        # 训练完成后构建 Faiss 索引
        self._build_faiss_index(num_items, device)

    def _build_faiss_index(self, num_items: int, device):
        logger.info("Building Faiss index for item embeddings...")
        self.model.eval()
        with torch.no_grad():
            all_item_ids = torch.arange(num_items).long().to(device)
            # 分批计算以防内存溢出
            item_embeddings = []
            for i in range(0, num_items, 1024):
                batch_ids = all_item_ids[i : i + 1024]
                item_embeddings.append(self.model.item_tower(batch_ids).cpu().numpy())
            
            item_vectors = np.vstack(item_embeddings).astype('float32')
            
            # 使用内积索引 (IndexFlatIP)
            self.faiss_index = faiss.IndexFlatIP(self.embed_dim)
            self.faiss_index.add(item_vectors)
            logger.success("Faiss index built.")

    def recall(self, user_id: int, top_n: int = 100) -> List[Tuple[int, float]]:
        """
        使用 Faiss 进行最近邻搜索。
        """
        if self.model is None or self.faiss_index is None:
            logger.error("Model or Faiss index not ready!")
            return []

        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id]).to(device)
            user_vec = self.model.user_tower(user_tensor).cpu().numpy().astype('float32')
            
            scores, indices = self.faiss_index.search(user_vec, top_n)
            
            return list(zip(indices[0].tolist(), scores[0].tolist()))

    def save(self, path: str) -> None:
        """保存模型权重和 Faiss 索引"""
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/model.pth")
        faiss.write_index(self.faiss_index, f"{path}/item_index.faiss")
        
        # 保存一些元数据
        meta = {'embed_dim': self.embed_dim, 'temp': self.temperature}
        with open(f"{path}/meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info(f"Two-Tower model saved to {path}")

    def load(self, path: str) -> None:
        # 实际加载需要根据训练时的 num_users/num_items 重新实例化模型
        # 这里为了演示先略过具体恢复逻辑，后续在 Pipeline 中完善
        logger.warning("Load logic needs exact num_users/num_items from metadata.")
