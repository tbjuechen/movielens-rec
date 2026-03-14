import torch
import torch.nn as pd
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=64):
        super().__init__()
        # Embeddings
        self.user_emb = nn.Embedding(vocab_sizes['userId'] + 1, embed_dim, padding_idx=0)
        self.item_emb = nn.Embedding(vocab_sizes['movieId'] + 1, embed_dim, padding_idx=0) # for history
        self.genre_emb = nn.Embedding(vocab_sizes['genres'] + 1, embed_dim, padding_idx=0)
        
        # DNN for continuous features (avg_rating, activity) -> scalar to vector
        self.continuous_dnn = nn.Linear(2, embed_dim)
        
        # Combine all features: user_id(1) + history_pool(1) + genres_pool(1) + continuous(1) = 4 * embed_dim
        input_dim = 4 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, features):
        u_emb = self.user_emb(features['user_id']) # (B, D)
        
        # History (Mean Pooling)
        hist_emb = self.item_emb(features['history']) # (B, SeqLen, D)
        # mask zero padding
        hist_mask = (features['history'] > 0).float().unsqueeze(-1)
        hist_emb = (hist_emb * hist_mask).sum(dim=1) / (hist_mask.sum(dim=1) + 1e-8) # (B, D)
        
        # Top Genres (Mean Pooling)
        genre_emb = self.genre_emb(features['top_genres']) # (B, 3, D)
        genre_mask = (features['top_genres'] > 0).float().unsqueeze(-1)
        genre_emb = (genre_emb * genre_mask).sum(dim=1) / (genre_mask.sum(dim=1) + 1e-8) # (B, D)
        
        # Continuous
        cont_feats = torch.stack([features['avg_rating'], features['activity']], dim=1) # (B, 2)
        cont_emb = F.relu(self.continuous_dnn(cont_feats)) # (B, D)
        
        # Concat & MLP
        concat_emb = torch.cat([u_emb, hist_emb, genre_emb, cont_emb], dim=1)
        out = self.mlp(concat_emb)
        return F.normalize(out, p=2, dim=1) # Normalize for inner product


class ItemTower(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=64):
        super().__init__()
        # Embeddings
        self.item_emb = nn.Embedding(vocab_sizes['movieId'] + 1, embed_dim, padding_idx=0)
        self.genre_emb = nn.Embedding(vocab_sizes['genres'] + 1, embed_dim, padding_idx=0)
        
        # Continuous (release_year, avg_rating, revenue)
        self.continuous_dnn = nn.Linear(3, embed_dim)
        
        # Combine: item_id(1) + genres_pool(1) + continuous(1) = 3 * embed_dim
        input_dim = 3 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, features):
        i_emb = self.item_emb(features['item_id'])
        
        # Genres (Mean Pooling)
        genre_emb = self.genre_emb(features['tmdb_genres'])
        genre_mask = (features['tmdb_genres'] > 0).float().unsqueeze(-1)
        genre_emb = (genre_emb * genre_mask).sum(dim=1) / (genre_mask.sum(dim=1) + 1e-8)
        
        # Continuous
        cont_feats = torch.stack([features['release_year'], features['avg_rating'], features['revenue']], dim=1)
        cont_emb = F.relu(self.continuous_dnn(cont_feats))
        
        concat_emb = torch.cat([i_emb, genre_emb, cont_emb], dim=1)
        out = self.mlp(concat_emb)
        return F.normalize(out, p=2, dim=1)

class DualTowerModel(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=64, tau=0.1):
        super().__init__()
        self.user_tower = UserTower(vocab_sizes, embed_dim)
        self.item_tower = ItemTower(vocab_sizes, embed_dim)
        self.tau = tau

    def forward(self, user_features, item_features):
        """
        用于推理：分别返回 user 和 item 的 embeddings
        """
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        return user_emb, item_emb

    def compute_loss(self, user_features, item_features, item_log_q, extra_item_features=None, extra_item_log_q=None):
        """
        用于训练：计算带有 Log-Q 纠偏的 InfoNCE Loss (In-batch + Global Negative Sampling)
        item_log_q: (B,) 正样本物品的采样概率的对数
        extra_item_log_q: (M,) 全局负样本物品的采样概率的对数
        """
        user_emb = self.user_tower(user_features) # (B, D)
        item_emb = self.item_tower(item_features) # (B, D)
        
        # 1. 计算 In-batch 相似度: (B, B)
        logits = torch.matmul(user_emb, item_emb.T) # (B, B)
        
        # 2. Log-Q 纠偏 (针对 In-batch 物品)
        # 减去 log(P(i))。注意：item_log_q 对应的是列方向上的物品概率
        # logits[i, j] 对应 user_i 和 item_j，因此减去的是 item_j 的 log_q
        logits = logits - item_log_q.view(1, -1)
        
        # 3. 如果有额外的全局负样本 (M, D)
        if extra_item_features is not None:
            extra_item_emb = self.item_tower(extra_item_features) # (M, D)
            extra_logits = torch.matmul(user_emb, extra_item_emb.T) # (B, M)
            
            # 对全局负样本也进行 Log-Q 纠偏
            if extra_item_log_q is not None:
                extra_logits = extra_logits - extra_item_log_q.view(1, -1)
                
            # 拼接列，得到 (B, B + M) 的 Logits
            logits = torch.cat([logits, extra_logits], dim=1)
            
        logits = logits / self.tau
        
        # Labels are still the diagonal indices (0, 1, ..., B-1)
        batch_size = user_emb.size(0)
        labels = torch.arange(batch_size, device=user_emb.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
