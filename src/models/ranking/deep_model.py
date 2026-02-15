import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import CrossNetV2, AttentionSequencePooling

class UnifiedDeepRanker(nn.Module):
    """
    高正则化精排模型：针对 99% 过拟合现象进行了强力约束。
    """
    def __init__(self, 
                 feature_map: dict, 
                 embedding_dim: int = 128,
                 hidden_units: list = [512, 256, 128],
                 dropout_rate: float = 0.2): # 恢复到 0.2
        super().__init__()
        self.feature_map = feature_map
        
        # 1. Embedding 层
        self.embeddings = nn.ModuleDict()
        for feat, size in feature_map['sparse'].items():
            self.embeddings[feat] = nn.Embedding(size + 1, embedding_dim, padding_idx=0)
            nn.init.normal_(self.embeddings[feat].weight, std=0.01)
            
        # 2. Sequence Layer (DIN)
        self.din_attention = AttentionSequencePooling(embedding_dim)
        
        # 3. 维度计算
        self.total_input_dim = (len(feature_map['sparse']) * embedding_dim) + \
                               embedding_dim + \
                               feature_map['dense']
        
        # 4. DCN Layer
        self.cross_net = CrossNetV2(self.total_input_dim, num_layers=2)
        
        # 5. Stability Layer
        self.input_bn = nn.BatchNorm1d(self.total_input_dim)
        
        # 6. Deep MLP Tower (带强力 Dropout)
        input_dim = self.total_input_dim
        curr_units = [input_dim] + hidden_units
        layers = []
        for i in range(len(curr_units) - 1):
            fc = nn.Linear(curr_units[i], curr_units[i+1])
            nn.init.kaiming_normal_(fc.weight, mode='fan_out', nonlinearity='leaky_relu')
            layers.append(fc)
            layers.append(nn.BatchNorm1d(curr_units[i+1]))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate)) # 0.5 的激进丢弃
        
        self.mlp_tower = nn.Sequential(*layers)
        
        # 7. Final Output Head
        self.click_head = nn.Linear(hidden_units[-1], 1)
        nn.init.xavier_normal_(self.click_head.weight)

    def forward(self, inputs):
        is_listwise = inputs['movieId'].dim() == 2
        if is_listwise:
            B, K = inputs['movieId'].shape
            flat_inputs = {
                'movieId': inputs['movieId'].reshape(-1),
                'seq_history': inputs['seq_history'].reshape(B*K, -1),
                'dense_feats': inputs['dense_feats'].reshape(B*K, -1)
            }
        else:
            flat_inputs = inputs
            B, K = inputs['movieId'].size(0), 1

        # A. Embeddings
        sparse_embs = [self.embeddings[feat](flat_inputs[feat]) for feat in self.feature_map['sparse']]
        
        # B. DIN
        target_item_emb = self.embeddings['movieId'](flat_inputs['movieId']).unsqueeze(1)
        history_seq_emb = self.embeddings['movieId'](flat_inputs['seq_history'])
        seq_mask = (flat_inputs['seq_history'] > 0).float()
        interest_emb = self.din_attention(target_item_emb, history_seq_emb, seq_mask)
        
        # C. Concatenate & BN
        all_features = torch.cat(sparse_embs + [interest_emb, flat_inputs['dense_feats']], dim=-1)
        all_features = self.input_bn(all_features)
        
        # D. DCN
        cross_out = self.cross_net(all_features)
        
        # E. MLP Tower
        deep_out = self.mlp_tower(cross_out)
        
        # F. Output Logits
        logits = self.click_head(deep_out).squeeze(-1)
        # 训练时使用温度缩放，推理时保持原始
        if self.training:
            logits = logits * 5.0 
        
        return {'click': logits}
