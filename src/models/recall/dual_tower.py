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

    def compute_loss(self, user_features, item_features, item_log_q, 
                     simple_neg_features=None, simple_neg_log_q=None,
                     hard_neg_features=None):
        """
        混合损失函数：
        1. InfoNCE: 处理 In-batch 和 随机负样本 (针对召回的全域分布优化)
        2. BPR: 处理 困难负样本 (针对正样本 vs 困难负样本的相对排序优化)
        """
        user_emb = self.user_tower(user_features) # (B, D)
        item_emb = self.item_tower(item_features) # (B, D)
        
        # --- 1. InfoNCE Loss (Simple Negatives) ---
        # 计算正样本得分 (对角线)
        pos_scores = torch.sum(user_emb * item_emb, dim=-1) # (B,)
        
        # In-batch 相似度矩阵 (B, B)
        logits = torch.matmul(user_emb, item_emb.T) 
        # Log-Q 纠偏
        logits = logits - item_log_q.view(1, -1)
        
        if simple_neg_features is not None:
            simple_neg_emb = self.item_tower(simple_neg_features) # (M1, D)
            simple_logits = torch.matmul(user_emb, simple_neg_emb.T) # (B, M1)
            if simple_neg_log_q is not None:
                simple_logits = simple_logits - simple_neg_log_q.view(1, -1)
            logits = torch.cat([logits, simple_logits], dim=1)
            
        logits = logits / self.tau
        batch_size = user_emb.size(0)
        labels = torch.arange(batch_size, device=user_emb.device)
        loss_infonce = F.cross_entropy(logits, labels)
        
        # --- 2. BPR Loss (Hard Negatives) ---
        loss_bpr = torch.tensor(0.0, device=user_emb.device)
        if hard_neg_features is not None:
            # 这里的 hard_neg_features 可能是全局热门物品 (M2, D)
            hard_neg_emb = self.item_tower(hard_neg_features) # (M2, D)
            # 计算每个 user 与所有 hard 负样本的得分 (B, M2)
            hard_scores = torch.matmul(user_emb, hard_neg_emb.T)
            
            # 计算 BPR: -log(sigmoid(pos_score - hard_score))
            # 我们取所有 hard 负样本的平均 BPR 贡献
            # pos_scores 是 (B, 1), hard_scores 是 (B, M2)
            diff = pos_scores.view(-1, 1) - hard_scores # (B, M2)
            loss_bpr = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
            
        # 混合权重建议 1.0 : 1.0，具体可调
        total_loss = loss_infonce + loss_bpr
        return total_loss, loss_infonce, loss_bpr
