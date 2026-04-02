import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config.settings import RANK_DIN_ATTENTION_HIDDEN, RANK_HIST_SEQ_MAXLEN
from src.models.ranking.modules import CrossNetV2, DINAttention, MMoE, TaskTower


def _quantile_bounds(values, n_buckets):
    """Compute quantile boundaries for continuous feature bucketization.

    Always returns exactly (n_buckets - 1) boundaries to ensure consistent
    buffer shapes across features and between training/evaluation.
    """
    q = np.linspace(0, 1, n_buckets + 1)[1:-1]
    bounds = np.quantile(values[~np.isnan(values)], q)
    return bounds.astype(np.float32)


class EmbeddingLayer(nn.Module):
    """Transforms base features and DIN history into mixed-dimension embeddings.

    ID features (user/item) → 64d embedding
    Genre features → 8d embedding (masked mean pooling)
    Continuous features → quantile bucketization → 8d embedding
    DIN history interest → 64d embedding
    """

    def __init__(self, vocab_sizes, id_embed_dim=64, genre_embed_dim=8,
                 cont_embed_dim=8, cont_bucket_size=20,
                 bucket_boundaries=None,
                 user_genre_max_len=10, item_genre_max_len=10,
                 dropout=0.1, hist_seq_maxlen=RANK_HIST_SEQ_MAXLEN):
        super().__init__()

        self.user_genre_len = user_genre_max_len
        self.item_genre_len = item_genre_max_len

        # 1-2. Sparse ID embeddings (64d)
        self.user_emb = nn.Embedding(vocab_sizes['userId'] + 1, id_embed_dim, padding_idx=0)
        self.item_emb = nn.Embedding(vocab_sizes['movieId'] + 1, id_embed_dim, padding_idx=0)
        self.din_attention = DINAttention(id_embed_dim, RANK_DIN_ATTENTION_HIDDEN, dropout=dropout)

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

        # History length bucket embedding (8d)
        self.hist_len_emb = nn.Embedding(cont_bucket_size, cont_embed_dim)
        hist_len_bounds = torch.linspace(0, float(hist_seq_maxlen), cont_bucket_size + 1)[1:-1]
        self.register_buffer('hist_len_bounds', hist_len_bounds)

        # Output dim: 2*id + 2*genre + 7*cont + hist_interest + target_hist_cross + hist_len
        self.output_dim = (2 * id_embed_dim + 2 * genre_embed_dim + 7 * cont_embed_dim
                           + id_embed_dim     # hist_interest
                           + id_embed_dim     # target_hist_cross
                           + cont_embed_dim)  # hist_len

    def forward(self, features):
        """Returns (B, output_dim) concatenated embedding vector."""
        # Unpack from merged blocks:
        # int_features: [user_id(0), item_id(1), user_genres(2:2+ug), item_genres(2+ug:2+ug+ig)]
        int_feat = features['int_features']
        float_feat = features['float_features']
        seq_feat = features['seq_features']
        
        embs = []

        # 1-2. Sparse IDs (64d each)
        user_emb = self.user_emb(int_feat[:, 0])
        item_emb = self.item_emb(int_feat[:, 1])
        embs.append(user_emb)
        embs.append(item_emb)

        # 3-4. Genre pooling — masked mean (8d each)
        ug_start = 2
        ug_end = ug_start + self.user_genre_len
        ig_end = ug_end + self.item_genre_len
        for col_start, col_end in ((ug_start, ug_end), (ug_end, ig_end)):
            g_ids = int_feat[:, col_start:col_end]
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

        hist_keys = self.item_emb(seq_feat)
        hist_mask = seq_feat > 0
        hist_interest = self.din_attention(item_emb, hist_keys, hist_mask)
        embs.append(hist_interest)
        embs.append(item_emb * hist_interest)  # target_hist_cross

        # History length bucket embedding
        hist_len = hist_mask.sum(dim=1).float()
        hist_len_bucket = torch.bucketize(hist_len, self.hist_len_bounds)
        embs.append(self.hist_len_emb(hist_len_bucket))

        return torch.cat(embs, dim=-1)  # (B, output_dim)


class RankingModel(torch.nn.Module):
    """DCNv2 + MMoE ranking model for a single CTR objective."""

    def __init__(self, vocab_sizes, id_embed_dim=64, genre_embed_dim=8,
                 cont_embed_dim=8, cont_bucket_size=20,
                 cross_layers=3, dropout=0.1,
                 num_experts=4, expert_dim=128,
                 tower_dims=None, bucket_boundaries=None,
                 user_genre_max_len=10, item_genre_max_len=10,
                 hist_seq_maxlen=RANK_HIST_SEQ_MAXLEN):
        super().__init__()
        if tower_dims is None:
            tower_dims = [64, 32]

        self.embedding_layer = EmbeddingLayer(
            vocab_sizes, id_embed_dim, genre_embed_dim, cont_embed_dim,
            cont_bucket_size, bucket_boundaries,
            user_genre_max_len, item_genre_max_len, dropout, hist_seq_maxlen
        )
        input_dim = self.embedding_layer.output_dim
        self.emb_dropout = nn.Dropout(dropout)

        self.cross_net = CrossNetV2(input_dim, cross_layers)
        self.mmoe = MMoE(input_dim, num_experts, expert_dim, num_tasks=1)

        tower_input_dim = input_dim + expert_dim  # cross_out + expert_out
        self.ctr_tower = TaskTower(tower_input_dim, tower_dims, use_sigmoid=False, dropout=dropout)

    def forward(self, features):
        emb = self.emb_dropout(self.embedding_layer(features))
        cross_out = self.cross_net(emb)
        ctr_expert = self.mmoe(emb)[0]
        return self.ctr_tower(torch.cat([cross_out, ctr_expert], dim=-1))

    def compute_loss(self, ctr_logit, ctr_label):
        """Returns the BCE loss for the CTR task."""
        return F.binary_cross_entropy_with_logits(ctr_logit, ctr_label, reduction='mean')
