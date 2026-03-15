import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, vocab_sizes, item_emb, genre_emb, embed_dim=64,
                 time_decay_lambda=0.001, cont_bucket_size=20, bucket_boundaries=None):
        super().__init__()
        self.user_emb = nn.Embedding(vocab_sizes['userId'] + 1, embed_dim, padding_idx=0)
        self.item_emb = item_emb
        self.genre_emb = genre_emb
        self.time_decay_lambda = time_decay_lambda
        self.avg_rating_bucket = nn.Embedding(cont_bucket_size, embed_dim)
        self.activity_bucket = nn.Embedding(cont_bucket_size, embed_dim)
        # Quantile-based boundaries (persisted in state_dict as buffers)
        default = torch.linspace(0, 1, cont_bucket_size + 1)[1:-1]
        if bucket_boundaries is not None:
            self.register_buffer('avg_rating_bounds', torch.as_tensor(bucket_boundaries['avg_rating'], dtype=torch.float32))
            self.register_buffer('activity_bounds', torch.as_tensor(bucket_boundaries['activity'], dtype=torch.float32))
        else:
            self.register_buffer('avg_rating_bounds', default.clone())
            self.register_buffer('activity_bounds', default.clone())

        self.mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embed_dim, embed_dim)
        )

    def forward(self, features):
        u_emb = self.user_emb(features['user_id'])

        hist_emb = self.item_emb(features['history'])
        hist_mask = (features['history'] > 0).float().unsqueeze(-1)
        delta_t = features.get('history_ts_diff', torch.zeros_like(features['history']).float())
        time_weights = torch.exp(-self.time_decay_lambda * delta_t)
        combined_weights = (time_weights * hist_mask.squeeze(-1)).unsqueeze(-1)
        hist_emb = (hist_emb * combined_weights).sum(dim=1) / (combined_weights.sum(dim=1) + 1e-8)

        genre_emb = self.genre_emb(features['top_genres'])
        genre_mask = (features['top_genres'] > 0).float().unsqueeze(-1)
        genre_emb = (genre_emb * genre_mask).sum(dim=1) / (genre_mask.sum(dim=1) + 1e-8)

        cont_emb = self.avg_rating_bucket(torch.bucketize(features['avg_rating'], self.avg_rating_bounds)) \
                  + self.activity_bucket(torch.bucketize(features['activity'], self.activity_bounds))

        concat_emb = torch.cat([u_emb, hist_emb, genre_emb, cont_emb], dim=1)
        out = self.mlp(concat_emb)
        return F.normalize(out, p=2, dim=1)

class ItemTower(nn.Module):
    def __init__(self, vocab_sizes, item_emb, genre_emb, embed_dim=64,
                 cont_bucket_size=20, bucket_boundaries=None):
        super().__init__()
        self.item_emb = item_emb
        self.genre_emb = genre_emb
        self.release_year_bucket = nn.Embedding(cont_bucket_size, embed_dim)
        self.avg_rating_bucket = nn.Embedding(cont_bucket_size, embed_dim)
        self.revenue_bucket = nn.Embedding(cont_bucket_size, embed_dim)
        # Quantile-based boundaries
        default = torch.linspace(0, 1, cont_bucket_size + 1)[1:-1]
        if bucket_boundaries is not None:
            self.register_buffer('release_year_bounds', torch.as_tensor(bucket_boundaries['release_year'], dtype=torch.float32))
            self.register_buffer('avg_rating_bounds', torch.as_tensor(bucket_boundaries['avg_rating'], dtype=torch.float32))
            self.register_buffer('revenue_bounds', torch.as_tensor(bucket_boundaries['revenue'], dtype=torch.float32))
        else:
            self.register_buffer('release_year_bounds', default.clone())
            self.register_buffer('avg_rating_bounds', default.clone())
            self.register_buffer('revenue_bounds', default.clone())

        self.mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embed_dim, embed_dim)
        )

    def forward(self, features):
        i_emb = self.item_emb(features['item_id'])
        genre_emb = self.genre_emb(features['tmdb_genres'])
        genre_mask = (features['tmdb_genres'] > 0).float().unsqueeze(-1)
        genre_emb = (genre_emb * genre_mask).sum(dim=1) / (genre_mask.sum(dim=1) + 1e-8)

        cont_emb = self.release_year_bucket(torch.bucketize(features['release_year'], self.release_year_bounds)) \
                  + self.avg_rating_bucket(torch.bucketize(features['avg_rating'], self.avg_rating_bounds)) \
                  + self.revenue_bucket(torch.bucketize(features['revenue'], self.revenue_bounds))

        concat_emb = torch.cat([i_emb, genre_emb, cont_emb], dim=1)
        out = self.mlp(concat_emb)
        return F.normalize(out, p=2, dim=1)

class DualTowerModel(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=64, tau=0.1,
                 inbatch_size=1024, global_size=512, hard_size=128,
                 time_decay_lambda=0.001, bpr_gamma=5.0, bpr_margin=0.1,
                 loss_infonce_weight=1.0, loss_bpr_weight=1.0,
                 logit_scale_max=100.0, cont_bucket_size=20,
                 user_bucket_boundaries=None, item_bucket_boundaries=None):
        super().__init__()
        self.item_emb = nn.Embedding(vocab_sizes['movieId'] + 1, embed_dim, padding_idx=0)
        self.genre_emb = nn.Embedding(vocab_sizes['genres'] + 1, embed_dim, padding_idx=0)
        self.user_tower = UserTower(vocab_sizes, self.item_emb, self.genre_emb, embed_dim,
                                    time_decay_lambda, cont_bucket_size, user_bucket_boundaries)
        self.item_tower = ItemTower(vocab_sizes, self.item_emb, self.genre_emb, embed_dim,
                                    cont_bucket_size, item_bucket_boundaries)
        # Learnable temperature: learn log(1/tau), clamp to prevent collapse
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / tau)))
        self.logit_scale_max = logit_scale_max
        self.bpr_gamma = bpr_gamma
        self.bpr_margin = bpr_margin
        self.loss_infonce_weight = loss_infonce_weight
        self.loss_bpr_weight = loss_bpr_weight

    def forward(self, user_features, item_features):
        u_emb = self.user_tower(user_features) if user_features is not None else None
        i_emb = self.item_tower(item_features) if item_features is not None else None
        return u_emb, i_emb

    def compute_loss(self, user_features, item_features, item_log_q,
                     inbatch_neg_emb, inbatch_neg_log_q,
                     global_neg_emb, global_neg_log_q,
                     hard_neg_emb,
                     collision_mask=None,
                     bpr_collision_mask=None):
        """
        collision_mask: (B, N_neg) InfoNCE 假负样本屏蔽
        bpr_collision_mask: (B, Hard_Size) BPR 假负样本屏蔽
        """
        user_emb = self.user_tower(user_features) # (B, D)
        item_emb = self.item_tower(item_features) # (B, D)

        # Learnable scale = 1/tau, clamped to prevent collapse
        scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)

        # 1. InfoNCE Logits
        pos_scores = torch.sum(user_emb * item_emb, dim=-1, keepdim=True) # (B, 1)
        pos_logits = (pos_scores * scale) - item_log_q.view(-1, 1)

        inbatch_logits = (torch.matmul(user_emb, inbatch_neg_emb.T) * scale) - inbatch_neg_log_q.view(1, -1)
        global_logits = (torch.matmul(user_emb, global_neg_emb.T) * scale) - global_neg_log_q.view(1, -1)
        
        neg_logits = torch.cat([inbatch_logits, global_logits], dim=1) # (B, N_neg)
        
        # Apply Collision Mask (Critical Fix #16)
        if collision_mask is not None:
            neg_logits = neg_logits.masked_fill(collision_mask, -1e9)
            
        infonce_logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(user_emb.size(0), dtype=torch.long, device=user_emb.device)
        loss_infonce = F.cross_entropy(infonce_logits, labels)
        
        # 2. BPR Loss (with margin)
        hard_scores = torch.matmul(user_emb, hard_neg_emb.T) # (B, Hard_Size)
        diff = (pos_scores.view(-1, 1) - hard_scores - self.bpr_margin) * self.bpr_gamma
        # Mask out false negatives: zero out diff so sigmoid(0)=0.5, no gradient signal
        if bpr_collision_mask is not None:
            diff = diff.masked_fill(bpr_collision_mask, 0.0)
        loss_bpr = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
        
        total_loss = self.loss_infonce_weight * loss_infonce + self.loss_bpr_weight * loss_bpr
        return total_loss, loss_infonce, loss_bpr
