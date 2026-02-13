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
from tqdm import tqdm
import sys
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

# --- 数据集类 V2 ---

class MovieLensDataset(Dataset):
    def __init__(self, df_interactions: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame):
        """
        df_interactions: 包含 userId, movieId, rating
        user_features: 包含 userId_int, user_avg_rating_norm, user_rating_count_norm, user_top_genres_idx
        item_features: 包含 movieId_int, genres_idx, release_year_norm, item_avg_rating_norm, item_rating_count_norm
        """
        self.users = df_interactions['userId'].values
        self.items = df_interactions['movieId'].values
        
        # 将特征表转换为字典，加速查询
        logger.info("Building feature lookup dictionaries...")
        self.user_feat_dict = user_features.set_index('userId_int').to_dict('index')
        self.item_feat_dict = item_features.set_index('movieId_int').to_dict('index')

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        mid = self.items[idx]
        
        u_feat = self.user_feat_dict[uid]
        i_feat = self.item_feat_dict[mid]
        
        return {
            "user_id": uid,
            "user_numeric": np.array([u_feat['user_avg_rating_norm'], u_feat['user_rating_count_norm']], dtype=np.float32),
            "user_genres": np.array(u_feat['user_top_genres_idx'], dtype=np.int64),
            "item_id": mid,
            "item_numeric": np.array([i_feat['release_year_norm'], i_feat['item_avg_rating_norm'], i_feat['item_rating_count_norm']], dtype=np.float32),
            "item_genres": np.array(i_feat['genres_idx'], dtype=np.int64)
        }

def collate_fn(batch):
    """
    自定义 Collate 函数，处理变长题材列表（进行零填充）。
    """
    res = {}
    for key in ["user_id", "item_id"]:
        res[key] = torch.LongTensor([x[key] for x in batch])
    
    for key in ["user_numeric", "item_numeric"]:
        res[key] = torch.FloatTensor(np.stack([x[key] for x in batch]))
        
    # 处理题材 Padding (user_genres 是固定长度 3，item_genres 是变长)
    res["user_genres"] = torch.LongTensor(np.stack([x["user_genres"] for x in batch]))
    
    item_genres_list = [torch.LongTensor(x["item_genres"]) for x in batch]
    res["item_genres"] = torch.nn.utils.rnn.pad_sequence(item_genres_list, batch_first=True, padding_value=0)
    
    return res

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

        logger.info("Data loading and tensor conversion finished. Starting training loop...")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", file=sys.stdout, mininterval=0.5)
            for user_ids, item_ids in pbar:
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
                
                current_loss = loss.item()
                total_loss += current_loss
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
            
            logger.info(f"Epoch {epoch+1}/{epochs} finished. Avg Loss: {total_loss/len(loader):.4f}")

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
        """从磁盘恢复模型、索引和元数据"""
        # 1. 加载元数据
        with open(f"{path}/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.embed_dim = meta['embed_dim']
        self.temperature = meta['temp']

        # 2. 恢复 Faiss 索引
        self.faiss_index = faiss.read_index(f"{path}/item_index.faiss")
        
        # 3. 恢复 PyTorch 模型
        # 注意：为了恢复模型，我们需要知道训练时的 num_users 和 num_items
        # 这里从 Faiss 索引和权重文件中推断
        state_dict = torch.load(f"{path}/model.pth", map_location="cpu")
        num_users = state_dict['user_tower.user_embed.weight'].shape[0]
        num_items = state_dict['item_tower.item_embed.weight'].shape[0]
        
        self.model = TwoTowerModel(num_users, num_items, self.embed_dim, self.temperature)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        logger.info(f"Two-Tower model loaded from {path} (Users: {num_users}, Items: {num_items})")
