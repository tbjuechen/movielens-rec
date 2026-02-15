import torch
import torch.nn as nn
from .layers import CrossNetV2, AttentionSequencePooling

class UnifiedDeepRanker(nn.Module):
    """
    DCN-V2 + DIN + Multi-Task Ranker
    集大成者：同时处理序列兴趣、高阶交叉和多目标预测。
    """
    def __init__(self, 
                 feature_map: dict, 
                 embedding_dim: int = 32,
                 hidden_units: list = [256, 128, 64],
                 tasks: list = ['click', 'rating']):
        super().__init__()
        self.feature_map = feature_map
        self.tasks = tasks
        
        # 1. Embedding Layers
        # 为每个稀疏特征建立 Embedding
        self.embeddings = nn.ModuleDict()
        for feat, size in feature_map['sparse'].items():
            self.embeddings[feat] = nn.Embedding(size + 1, embedding_dim, padding_idx=0)
            
        # 2. Sequence Layer (DIN)
        # 假设序列也是由 MovieID 组成，共享 MovieID 的 Embedding
        self.din_attention = AttentionSequencePooling(embedding_dim)
        
        # 3. Input Dimension Calculation
        # Total Dim = (Sparse Feats * EmbDim) + (Interest Vector * EmbDim) + Dense Feats
        self.total_input_dim = (len(feature_map['sparse']) * embedding_dim) + 
                               embedding_dim + 
                               feature_map['dense']
        
        # 4. DCN Layer
        self.cross_net = CrossNetV2(self.total_input_dim, num_layers=3)
        
        # 5. Deep Tower (Shared)
        input_dim = self.total_input_dim
        self.deep_layers = nn.ModuleList()
        for unit in hidden_units:
            self.deep_layers.append(nn.Linear(input_dim, unit))
            self.deep_layers.append(nn.BatchNorm1d(unit))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(0.1))
            input_dim = unit
            
        # 6. Multi-Task Heads
        self.heads = nn.ModuleDict()
        for task in tasks:
            if task == 'rating':
                # 回归任务 (1-5分)
                self.heads[task] = nn.Linear(input_dim, 1)
            else:
                # 二分类任务 (Sigmoid 在 Loss 中处理或推理时处理)
                self.heads[task] = nn.Linear(input_dim, 1)

    def forward(self, inputs):
        """
        inputs: Dict of Tensors
        - sparse_feats: {feat_name: (B,)}
        - dense_feats: (B, D)
        - seq_feats: (B, T)
        """
        # A. Sparse Embeddings
        sparse_embs = []
        for feat, layer in self.embeddings.items():
            sparse_embs.append(layer(inputs[feat])) # (B, E)
        
        # B. DIN Processing
        # Target Item (MovieID) Embedding
        target_item_emb = self.embeddings['movieId'](inputs['movieId']).unsqueeze(1) # (B, 1, E)
        # History Sequence Embedding
        history_seq_emb = self.embeddings['movieId'](inputs['seq_history']) # (B, T, E)
        # Mask (假设 0 是 padding)
        seq_mask = (inputs['seq_history'] > 0).float()
        
        interest_emb = self.din_attention(target_item_emb, history_seq_emb, seq_mask) # (B, E)
        
        # C. Concatenation
        # [All Sparse Embs, Interest Emb, Dense Feats]
        all_features = torch.cat(sparse_embs + [interest_emb, inputs['dense_feats']], dim=1)
        
        # D. DCN & Deep
        cross_out = self.cross_net(all_features)
        deep_out = cross_out
        for layer in self.deep_layers:
            deep_out = layer(deep_out)
            
        # E. Multi-Heads
        outputs = {}
        for task, head in self.heads.items():
            out = head(deep_out)
            # 输出 Logits，由 Loss 函数处理 Sigmoid
            outputs[task] = out.squeeze(1)
            
        return outputs
