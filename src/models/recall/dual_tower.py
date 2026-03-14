import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, vocab_sizes, item_emb, genre_emb, embed_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(vocab_sizes['userId'] + 1, embed_dim, padding_idx=0)
        self.item_emb = item_emb
        self.genre_emb = genre_emb
        self.continuous_dnn = nn.Linear(2, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, features):
        u_emb = self.user_emb(features['user_id'])
        
        hist_emb = self.item_emb(features['history'])
        hist_mask = (features['history'] > 0).float().unsqueeze(-1)
        # Use history_ts_diff for weighting
        delta_t = features.get('history_ts_diff', torch.zeros_like(features['history']).float())
        time_weights = torch.exp(-0.001 * delta_t) 
        combined_weights = (time_weights * hist_mask.squeeze(-1)).unsqueeze(-1)
        hist_emb = (hist_emb * combined_weights).sum(dim=1) / (combined_weights.sum(dim=1) + 1e-8)
        
        genre_emb = self.genre_emb(features['top_genres'])
        genre_mask = (features['top_genres'] > 0).float().unsqueeze(-1)
        genre_emb = (genre_emb * genre_mask).sum(dim=1) / (genre_mask.sum(dim=1) + 1e-8)
        
        cont_feats = torch.stack([features['avg_rating'], features['activity']], dim=1)
        cont_emb = F.relu(self.continuous_dnn(cont_feats))
        
        concat_emb = torch.cat([u_emb, hist_emb, genre_emb, cont_emb], dim=1)
        out = self.mlp(concat_emb)
        return F.normalize(out, p=2, dim=1)

class ItemTower(nn.Module):
    def __init__(self, vocab_sizes, item_emb, genre_emb, embed_dim=64):
        super().__init__()
        self.item_emb = item_emb
        self.genre_emb = genre_emb
        self.continuous_dnn = nn.Linear(3, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, features):
        i_emb = self.item_emb(features['item_id'])
        genre_emb = self.genre_emb(features['tmdb_genres'])
        genre_mask = (features['tmdb_genres'] > 0).float().unsqueeze(-1)
        genre_emb = (genre_emb * genre_mask).sum(dim=1) / (genre_mask.sum(dim=1) + 1e-8)
        
        cont_feats = torch.stack([features['release_year'], features['avg_rating'], features['revenue']], dim=1)
        cont_emb = F.relu(self.continuous_dnn(cont_feats))
        
        concat_emb = torch.cat([i_emb, genre_emb, cont_emb], dim=1)
        out = self.mlp(concat_emb)
        return F.normalize(out, p=2, dim=1)

class DualTowerModel(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=64, tau=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(vocab_sizes['movieId'] + 1, embed_dim, padding_idx=0)
        self.genre_emb = nn.Embedding(vocab_sizes['genres'] + 1, embed_dim, padding_idx=0)
        self.user_tower = UserTower(vocab_sizes, self.item_emb, self.genre_emb, embed_dim)
        self.item_tower = ItemTower(vocab_sizes, self.item_emb, self.genre_emb, embed_dim)
        self.tau = tau

    def forward(self, user_features, item_features):
        return self.user_tower(user_features), self.item_tower(item_features)

    def compute_loss(self, user_features, item_features, item_log_q, 
                     inbatch_neg_emb, inbatch_neg_log_q,
                     global_neg_emb, global_neg_log_q,
                     hard_neg_emb):
        """
        计算损失：固定负样本规模
        inbatch_neg_emb: (1024, D)
        global_neg_emb: (512, D)
        hard_neg_emb: (128, D)
        """
        user_emb = self.user_tower(user_features) # (B, D)
        item_emb = self.item_tower(item_features) # (B, D)
        
        # 1. 正样本得分
        pos_logits = torch.sum(user_emb * item_emb, dim=-1, keepdim=True) # (B, 1)
        pos_logits = (pos_logits / self.tau) - item_log_q.view(-1, 1)
        
        # 2. In-batch 风格负样本得分 (1024个)
        inbatch_logits = torch.matmul(user_emb, inbatch_neg_emb.T) # (B, 1024)
        inbatch_logits = (inbatch_logits / self.tau) - inbatch_neg_log_q.view(1, -1)
        
        # 3. 全局随机负样本得分 (512个)
        global_logits = torch.matmul(user_emb, global_neg_emb.T) # (B, 512)
        global_logits = (global_logits / self.tau) - global_neg_log_q.view(1, -1)
        
        # 组装 InfoNCE Logits
        infonce_logits = torch.cat([pos_logits, inbatch_logits, global_logits], dim=1) # (B, 1+1024+512)
        
        # Target 是第 0 列
        labels = torch.zeros(user_emb.size(0), dtype=torch.long, device=user_emb.device)
        loss_infonce = F.cross_entropy(infonce_logits, labels)
        
        # 4. BPR 损失 (针对 128 个困难负样本)
        hard_logits = torch.matmul(user_emb, hard_neg_emb.T) # (B, 128)
        # BPR: -log(sigmoid(pos - hard))
        diff = (pos_logits + item_log_q.view(-1, 1)) - hard_logits # 移除 logq 纠偏进行纯排序对比
        loss_bpr = -torch.log(torch.sigmoid(diff / self.tau) + 1e-8).mean()
        
        return loss_infonce + loss_bpr, loss_infonce, loss_bpr
