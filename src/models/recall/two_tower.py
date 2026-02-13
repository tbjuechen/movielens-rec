import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import faiss
import sys
from typing import List, Tuple, Dict
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.models.recall.base import BaseRecall

# --- PyTorch 模型定义 V2 ---

class UserTower(nn.Module):
    def __init__(self, num_users: int, num_genres: int, embed_dim: int = 128, genre_embed_dim: int = 16):
        super(UserTower, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        # MPS 兼容：使用标准 Embedding + 手动 mean 替代 EmbeddingBag
        self.genre_embed = nn.Embedding(num_genres + 1, genre_embed_dim, padding_idx=0)
        
        input_dim = embed_dim + genre_embed_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

    def forward(self, user_ids, genre_ids, numeric_feats):
        u_emb = self.user_embed(user_ids)
        g_emb = self.genre_embed(genre_ids).mean(dim=1)
        combined = torch.cat([u_emb, g_emb, numeric_feats], dim=1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=1)

class ItemTower(nn.Module):
    def __init__(self, num_items: int, num_genres: int, embed_dim: int = 128, genre_embed_dim: int = 16):
        super(ItemTower, self).__init__()
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.genre_embed = nn.Embedding(num_genres + 1, genre_embed_dim, padding_idx=0)
        
        input_dim = embed_dim + genre_embed_dim + 3
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

    def forward(self, item_ids, genre_ids, numeric_feats):
        i_emb = self.item_embed(item_ids)
        g_emb = self.genre_embed(genre_ids).mean(dim=1)
        combined = torch.cat([i_emb, g_emb, numeric_feats], dim=1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres, embed_dim=128, temperature=0.07):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(num_users, num_genres, embed_dim)
        self.item_tower = ItemTower(num_items, num_genres, embed_dim)
        self.temperature = temperature

    def forward(self, user_inputs, item_inputs):
        user_vec = self.user_tower(*user_inputs)
        item_vec = self.item_tower(*item_inputs)
        return user_vec, item_vec

# --- 数据集与加载辅助 ---

class MovieLensDataset(Dataset):
    def __init__(self, df_interactions, user_features, item_features):
        self.users = df_interactions['userId'].values
        self.items = df_interactions['movieId'].values
        self.user_feat_dict = user_features.set_index('userId_int').to_dict('index')
        self.item_feat_dict = item_features.set_index('movieId_int').to_dict('index')

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid, mid = self.users[idx], self.items[idx]
        u_f, i_f = self.user_feat_dict[uid], self.item_feat_dict[mid]
        return {
            "user_id": uid,
            "user_genres": np.array(u_f['user_top_genres_idx'], dtype=np.int64),
            "user_numeric": np.array([u_f['user_avg_rating_norm'], u_f['user_rating_count_norm']], dtype=np.float32),
            "item_id": mid,
            "item_genres": np.array(i_f['genres_idx'], dtype=np.int64),
            "item_numeric": np.array([i_f['release_year_norm'], i_f['item_avg_rating_norm'], i_f['item_rating_count_norm']], dtype=np.float32)
        }

def collate_fn(batch):
    def get_tensor(key, dtype):
        if key in ["user_numeric", "item_numeric"]:
            return torch.tensor(np.stack([x[key] for x in batch]), dtype=dtype)
        return torch.tensor([x[key] for x in batch], dtype=dtype)

    user_ids = get_tensor("user_id", torch.long)
    user_genres_list = [torch.tensor(x["user_genres"]) for x in batch]
    user_genres = torch.nn.utils.rnn.pad_sequence(user_genres_list, batch_first=True, padding_value=0)
    user_num = get_tensor("user_numeric", torch.float)
    
    item_ids = get_tensor("item_id", torch.long)
    item_genres_list = [torch.tensor(x["item_genres"]) for x in batch]
    item_genres = torch.nn.utils.rnn.pad_sequence(item_genres_list, batch_first=True, padding_value=0)
    item_num = get_tensor("item_numeric", torch.float)
    
    return (user_ids, user_genres, user_num), (item_ids, item_genres, item_num)

# --- 召回封装类 V2 ---

class TwoTowerRecall(BaseRecall):
    def __init__(self, embed_dim: int = 128, temperature: float = 0.07):
        super().__init__(name="TwoTowerRecall_V2")
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.model = None
        self.faiss_index = None
        self.user_map = {}
        self.movie_map = {}
        self.inv_movie_map = {}
        self.user_features_dict = {}
        self.item_feat_table = None

    def train(self, df_train: pd.DataFrame, **kwargs) -> None:
        user_features = kwargs.get('user_features')
        item_features = kwargs.get('item_features')
        self.user_map = kwargs.get('user_map', {})
        self.movie_map = kwargs.get('movie_map', {})
        self.inv_movie_map = {v: k for k, v in self.movie_map.items()}
        self.item_feat_table = item_features
        self.user_features_dict = user_features.set_index('userId_int').to_dict('index')
        
        num_users = user_features['userId_int'].max() + 1
        num_items = item_features['movieId_int'].max() + 1
        num_genres = 20
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = TwoTowerModel(num_users, num_items, num_genres, self.embed_dim, self.temperature).to(device)
        
        dataset = MovieLensDataset(df_train, user_features, item_features)
        loader = DataLoader(dataset, batch_size=kwargs.get('batch_size', 1024), shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs.get('lr', 0.001))

        logger.info(f"Training V2 on {device}...")
        self.model.train()
        for epoch in range(kwargs.get('epochs', 5)):
            total_loss, pbar = 0, tqdm(loader, desc=f"Epoch {epoch+1}", file=sys.stdout, mininterval=0.5)
            for user_in, item_in in pbar:
                user_in = [t.to(device) for t in user_in]
                item_in = [t.to(device) for t in item_in]
                u_vec, i_vec = self.model(user_in, item_in)
                logits = torch.matmul(u_vec, i_vec.T) / self.temperature
                loss = F.cross_entropy(logits, torch.arange(u_vec.size(0)).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            logger.info(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader):.4f}")

        self._build_faiss_index(device)

    def _build_faiss_index(self, device):
        logger.info("Building Faiss index...")
        self.model.eval()
        item_ids = torch.arange(len(self.item_feat_table)).to(device)
        item_genres_list = [torch.tensor(x) for x in self.item_feat_table['genres_idx'].values]
        item_genres = torch.nn.utils.rnn.pad_sequence(item_genres_list, batch_first=True).to(device)
        item_num = torch.tensor(np.stack(self.item_feat_table[['release_year_norm', 'item_avg_rating_norm', 'item_rating_count_norm']].values), dtype=torch.float).to(device)
        
        with torch.no_grad():
            embeddings = []
            for i in range(0, len(item_ids), 1024):
                embeddings.append(self.model.item_tower(item_ids[i:i+1024], item_genres[i:i+1024], item_num[i:i+1024]).cpu().numpy())
            self.faiss_index = faiss.IndexFlatIP(self.embed_dim)
            self.faiss_index.add(np.vstack(embeddings).astype('float32'))
        logger.success("Faiss index ready.")

    def recall(self, user_id: int, top_n: int = 100) -> List[Tuple[int, float]]:
        if user_id not in self.user_map: return []
        uid_int = self.user_map[user_id]
        feat = self.user_features_dict.get(uid_int)
        if not feat: return []

        device = next(self.model.parameters()).device
        u_id_tensor = torch.LongTensor([uid_int]).to(device)
        
        # 修复：对题材列表进行 Padding 到长度 3
        genres = feat['user_top_genres_idx']
        padded_genres = genres + [0] * (3 - len(genres))
        u_genres_tensor = torch.LongTensor([padded_genres]).to(device)
        
        u_num_tensor = torch.FloatTensor([[feat['user_avg_rating_norm'], feat['user_rating_count_norm']]]).to(device)

        self.model.eval()
        with torch.no_grad():
            user_vec = self.model.user_tower(u_id_tensor, u_genres_tensor, u_num_tensor).cpu().numpy().astype('float32')
            scores, indices = self.faiss_index.search(user_vec, top_n)
            return [(self.inv_movie_map[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx in self.inv_movie_map]

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/model.pth")
        faiss.write_index(self.faiss_index, f"{path}/item_index.faiss")
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump({'meta': {'embed_dim': self.embed_dim, 'temp': self.temperature}, 'user_map': self.user_map, 'movie_map': self.movie_map, 'user_features_dict': self.user_features_dict}, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        with open(f"{path}/metadata.pkl", "rb") as f:
            data = pickle.load(f)
        self.user_map, self.movie_map, self.user_features_dict = data['user_map'], data['movie_map'], data['user_features_dict']
        self.inv_movie_map = {v: k for k, v in self.movie_map.items()}
        self.faiss_index = faiss.read_index(f"{path}/item_index.faiss")
        
        # 自动识别设备
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        state_dict = torch.load(f"{path}/model.pth", map_location=device)
        num_users = state_dict['user_tower.user_embed.weight'].shape[0]
        num_items = state_dict['item_tower.item_embed.weight'].shape[0]
        self.model = TwoTowerModel(num_users, num_items, 20, data['meta']['embed_dim'], data['meta']['temp']).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Model loaded from {path} on {device}")
