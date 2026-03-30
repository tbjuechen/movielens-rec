import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.ranking.modules import CrossNetV2, MMoE, TaskTower


def _quantile_bounds(values, n_buckets):
    """Compute quantile boundaries for continuous feature bucketization.

    Always returns exactly (n_buckets - 1) boundaries to ensure consistent
    buffer shapes across features and between training/evaluation.
    """
    q = np.linspace(0, 1, n_buckets + 1)[1:-1]
    bounds = np.quantile(values[~np.isnan(values)], q)
    return bounds.astype(np.float32)


class EmbeddingLayer(nn.Module):
    """Transforms 11 feature fields into mixed-dimension embeddings.

    ID features (user/item) → 64d embedding
    Genre features → 8d embedding (masked mean pooling)
    Continuous features → quantile bucketization → 8d embedding
    """

    def __init__(self, vocab_sizes, id_embed_dim=64, genre_embed_dim=8,
                 cont_embed_dim=8, cont_bucket_size=20,
                 bucket_boundaries=None):
        super().__init__()

        # 1-2. Sparse ID embeddings (64d)
        self.user_emb = nn.Embedding(vocab_sizes['userId'] + 1, id_embed_dim, padding_idx=0)
        self.item_emb = nn.Embedding(vocab_sizes['movieId'] + 1, id_embed_dim, padding_idx=0)

        # 3-4. Genre embeddings (8d)
        self.genre_emb = nn.Embedding(vocab_sizes['genres'] + 1, genre_embed_dim, padding_idx=0)

        # 5-11. Continuous bucket embeddings (8d)
        cont_features = [
            'user_avg_rating', 'user_activity',
            'item_release_year', 'item_avg_rating', 'item_revenue',
            'item_budget', 'item_vote_count',
        ]
        self.bucket_embs = nn.ModuleDict({
            name: nn.Embedding(cont_bucket_size, cont_embed_dim)
            for name in cont_features
        })

        # Register bucket boundaries as buffers
        default_bounds = torch.linspace(0, 1, cont_bucket_size + 1)[1:-1]
        for name in cont_features:
            if bucket_boundaries and name in bucket_boundaries:
                self.register_buffer(
                    f'{name}_bounds',
                    torch.as_tensor(bucket_boundaries[name], dtype=torch.float32)
                )
            else:
                self.register_buffer(f'{name}_bounds', default_bounds.clone())

        # Output dim: 2*id + 2*genre + 7*cont
        self.output_dim = 2 * id_embed_dim + 2 * genre_embed_dim + 7 * cont_embed_dim

    def forward(self, features):
        """Returns (B, output_dim) concatenated embedding vector."""
        # Unpack from merged blocks: 
        # int_features: [user_id(0), item_id(1), user_genres(2:12), item_genres(12:22)]
        int_feat = features['int_features']
        float_feat = features['float_features']
        
        embs = []

        # 1-2. Sparse IDs (64d each)
        embs.append(self.user_emb(int_feat[:, 0]))
        embs.append(self.item_emb(int_feat[:, 1]))

        # 3-4. Genre pooling — masked mean (8d each)
        # user_genres: 2 to 12, item_genres: 12 to 22
        for col_start in (2, 12):
            g_ids = int_feat[:, col_start : col_start + 10]
            g_emb = self.genre_emb(g_ids)
            mask = (g_ids > 0).float().unsqueeze(-1)
            pooled = (g_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            embs.append(pooled)

        # 5-11. Continuous features — bucketized from float block (8d each)
        cont_keys = [
            'user_avg_rating', 'user_activity',
            'item_release_year', 'item_avg_rating', 'item_revenue',
            'item_budget', 'item_vote_count',
        ]
        # float_feat has shape (B, 7)
        for i, name in enumerate(cont_keys):
            bounds = getattr(self, f'{name}_bounds')
            # Slice [:, i] is non-contiguous, need .contiguous() for searchsorted performance
            val = float_feat[:, i].contiguous()
            bucket_idx = torch.bucketize(val, bounds)
            embs.append(self.bucket_embs[name](bucket_idx))

        return torch.cat(embs, dim=-1)  # (B, output_dim)


class RankingModel(nn.Module):
    """DCNv2 + MMoE ranking model with dual objectives (pCTR + pRating)."""

    def __init__(self, vocab_sizes, id_embed_dim=64, genre_embed_dim=8,
                 cont_embed_dim=8, cont_bucket_size=20,
                 cross_layers=3, dropout=0.1,
                 num_experts=4, expert_dim=128,
                 tower_dims=None, bucket_boundaries=None):
        super().__init__()
        if tower_dims is None:
            tower_dims = [64, 32]

        self.embedding_layer = EmbeddingLayer(
            vocab_sizes, id_embed_dim, genre_embed_dim, cont_embed_dim,
            cont_bucket_size, bucket_boundaries
        )
        input_dim = self.embedding_layer.output_dim

        self.cross_net = CrossNetV2(input_dim, cross_layers)
        self.mmoe = MMoE(input_dim, num_experts, expert_dim, num_tasks=2)

        tower_input_dim = input_dim + expert_dim  # cross_out + expert_out
        self.ctr_tower = TaskTower(tower_input_dim, tower_dims, use_sigmoid=False)
        self.rating_tower = TaskTower(tower_input_dim, tower_dims, use_sigmoid=False)

    def forward(self, features):
        emb = self.embedding_layer(features)       # (B, input_dim)
        cross_out = self.cross_net(emb)             # (B, input_dim)
        ctr_expert, rat_expert = self.mmoe(emb)     # (B, expert_dim) each

        ctr_logit = self.ctr_tower(torch.cat([cross_out, ctr_expert], dim=-1))
        pRating = self.rating_tower(torch.cat([cross_out, rat_expert], dim=-1))
        return ctr_logit, pRating

    def compute_loss(self, ctr_logit, pRating, ctr_label, rating_label, has_rating):
        """Returns individual task losses (no weighting — handled by GradNorm externally)."""
        # 1. BCE with logits for CTR (AMP-safe, numerically stable)
        loss_bce = F.binary_cross_entropy_with_logits(ctr_logit, ctr_label, reduction='mean')

        # 2. MSE for Rating (masked: only on interacted items with real ratings)
        loss_mse = torch.tensor(0.0, device=pRating.device)
        if has_rating.any():
            loss_mse = F.mse_loss(pRating[has_rating], rating_label[has_rating])

        return loss_bce, loss_mse
