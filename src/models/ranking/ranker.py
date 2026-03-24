import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.ranking.modules import CrossNetV2, MMoE, TaskTower


def _quantile_bounds(values, n_buckets):
    """Compute quantile boundaries for continuous feature bucketization."""
    q = np.linspace(0, 1, n_buckets + 1)[1:-1]
    bounds = np.quantile(values[~np.isnan(values)], q)
    return np.unique(bounds).astype(np.float32)


class EmbeddingLayer(nn.Module):
    """Transforms 14 feature fields into mixed-dimension embeddings.

    ID features (user/item) → 64d embedding
    Genre features → 8d embedding (masked mean pooling)
    Continuous features → quantile bucketization → 8d embedding
    Pre-trained dense → 128d direct pass-through (no projection)
    """

    def __init__(self, vocab_sizes, id_embed_dim=64, genre_embed_dim=8,
                 cont_embed_dim=8, cont_bucket_size=20,
                 pretrained_emb_dim=128, bucket_boundaries=None):
        super().__init__()
        self.pretrained_emb_dim = pretrained_emb_dim

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
        # 14. recall_sim_score also gets bucketized (8d)
        self.bucket_embs['recall_sim_score'] = nn.Embedding(cont_bucket_size, cont_embed_dim)

        # Register bucket boundaries as buffers
        default_bounds = torch.linspace(0, 1, cont_bucket_size + 1)[1:-1]
        for name in cont_features + ['recall_sim_score']:
            if bucket_boundaries and name in bucket_boundaries:
                self.register_buffer(
                    f'{name}_bounds',
                    torch.as_tensor(bucket_boundaries[name], dtype=torch.float32)
                )
            else:
                self.register_buffer(f'{name}_bounds', default_bounds.clone())

        # Output dim: 2*id + 2*genre + 8*cont + 2*pretrained
        self.output_dim = (2 * id_embed_dim + 2 * genre_embed_dim
                           + 8 * cont_embed_dim + 2 * pretrained_emb_dim)

    def forward(self, features):
        """Returns (B, output_dim) concatenated embedding vector."""
        embs = []

        # 1-2. Sparse IDs (64d each)
        embs.append(self.user_emb(features['user_id']))
        embs.append(self.item_emb(features['item_id']))

        # 3-4. Genre pooling — masked mean (8d each)
        for key in ('user_top_genres', 'item_genres'):
            g_emb = self.genre_emb(features[key])
            mask = (features[key] > 0).float().unsqueeze(-1)
            pooled = (g_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            embs.append(pooled)

        # 5-11. Continuous features — bucketized (8d each)
        cont_keys = [
            'user_avg_rating', 'user_activity',
            'item_release_year', 'item_avg_rating', 'item_revenue',
            'item_budget', 'item_vote_count',
        ]
        for name in cont_keys:
            bounds = getattr(self, f'{name}_bounds')
            bucket_idx = torch.bucketize(features[name], bounds)
            embs.append(self.bucket_embs[name](bucket_idx))

        # 12-13. Pre-trained embeddings — direct pass-through (128d each)
        embs.append(features['user_emb_pretrained'])
        embs.append(features['item_emb_pretrained'])

        # 14. Recall similarity score — bucketized (8d)
        sim_score = (features['user_emb_pretrained'] * features['item_emb_pretrained']).sum(dim=-1)
        sim_score = sim_score / self.pretrained_emb_dim
        bounds = getattr(self, 'recall_sim_score_bounds')
        sim_bucket = torch.bucketize(sim_score, bounds)
        embs.append(self.bucket_embs['recall_sim_score'](sim_bucket))

        return torch.cat(embs, dim=-1)  # (B, output_dim)


class RankingModel(nn.Module):
    """DCNv2 + MMoE ranking model with dual objectives (pCTR + pRating)."""

    def __init__(self, vocab_sizes, id_embed_dim=64, genre_embed_dim=8,
                 cont_embed_dim=8, cont_bucket_size=20,
                 pretrained_emb_dim=128, cross_layers=3,
                 dropout=0.1,
                 num_experts=4, expert_dim=128,
                 tower_dims=None, bucket_boundaries=None):
        super().__init__()
        if tower_dims is None:
            tower_dims = [64, 32]

        self.embedding_layer = EmbeddingLayer(
            vocab_sizes, id_embed_dim, genre_embed_dim, cont_embed_dim,
            cont_bucket_size, pretrained_emb_dim, bucket_boundaries
        )
        input_dim = self.embedding_layer.output_dim

        self.cross_net = CrossNetV2(input_dim, cross_layers)
        self.mmoe = MMoE(input_dim, num_experts, expert_dim, num_tasks=2)

        tower_input_dim = input_dim + expert_dim  # cross_out + expert_out
        self.ctr_tower = TaskTower(tower_input_dim, tower_dims, use_sigmoid=True)
        self.rating_tower = TaskTower(tower_input_dim, tower_dims, use_sigmoid=False)

    def forward(self, features):
        emb = self.embedding_layer(features)       # (B, input_dim)
        cross_out = self.cross_net(emb)             # (B, input_dim)
        ctr_expert, rat_expert = self.mmoe(emb)     # (B, expert_dim) each

        pCTR = self.ctr_tower(torch.cat([cross_out, ctr_expert], dim=-1))
        pRating = self.rating_tower(torch.cat([cross_out, rat_expert], dim=-1))
        return pCTR, pRating

    def compute_loss(self, pCTR, pRating, ctr_label, rating_label, has_rating,
                     ctr_bce_weight=1.0, ctr_bpr_weight=0.3, rating_mse_weight=0.5):
        # 1. BCE for CTR
        loss_bce = F.binary_cross_entropy(pCTR, ctr_label, reduction='mean')

        # 2. BPR auxiliary: batch-hard pairwise ranking
        # Each positive pairs with the hardest negative (highest-scored neg in batch)
        loss_bpr = torch.tensor(0.0, device=pCTR.device)
        pos_mask = ctr_label > 0.5
        neg_mask = ~pos_mask
        if pos_mask.any() and neg_mask.any():
            pos_scores = pCTR[pos_mask]
            neg_scores = pCTR[neg_mask]
            # Sort negatives descending → hardest first
            hard_neg_sorted = neg_scores.sort(descending=True).values
            n_pos = len(pos_scores)
            n_neg = len(hard_neg_sorted)
            # Repeat hard negatives if fewer than positives
            if n_neg < n_pos:
                repeats = (n_pos + n_neg - 1) // n_neg
                hard_neg_sorted = hard_neg_sorted.repeat(repeats)
            loss_bpr = -F.logsigmoid(pos_scores - hard_neg_sorted[:n_pos]).mean()

        # 3. MSE for Rating (masked: only on interacted items with real ratings)
        loss_mse = torch.tensor(0.0, device=pRating.device)
        if has_rating.any():
            loss_mse = F.mse_loss(pRating[has_rating], rating_label[has_rating])

        total = ctr_bce_weight * loss_bce + ctr_bpr_weight * loss_bpr + rating_mse_weight * loss_mse
        return total, loss_bce, loss_bpr, loss_mse
