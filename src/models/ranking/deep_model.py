import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import CrossNetV2, AttentionSequencePooling

class MMoELayer(nn.Module):
    """
    MMoE (Multi-gate Mixture-of-Experts) 核心层
    """
    def __init__(self, input_dim, num_experts, expert_hidden_units, num_tasks):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # 1. Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_units[0]),
                nn.ReLU(),
                nn.Linear(expert_hidden_units[0], expert_hidden_units[1])
            ) for _ in range(num_experts)
        ])
        
        # 2. Gates
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0) # (num_experts, B, expert_dim)
        
        final_outputs = []
        for i in range(self.num_tasks):
            gate_weights = self.gates[i](x) # (B, num_experts)
            # Transpose weights to (num_experts, B, 1) for broadcasting
            weights = gate_weights.t().unsqueeze(-1)
            task_out = torch.sum(weights * expert_outputs, dim=0) # (B, expert_dim)
            final_outputs.append(task_out)
            
        return final_outputs

class UnifiedDeepRanker(nn.Module):
    def __init__(self, 
                 feature_map: dict, 
                 embedding_dim: int = 128,
                 num_experts: int = 4,
                 tasks: list = ['click', 'rating']):
        super().__init__()
        self.feature_map = feature_map
        self.tasks = tasks
        
        # 1. Embedding Layers
        self.embeddings = nn.ModuleDict()
        for feat, size in feature_map['sparse'].items():
            self.embeddings[feat] = nn.Embedding(size + 1, embedding_dim, padding_idx=0)
            
        # 2. Sequence Layer (DIN)
        self.din_attention = AttentionSequencePooling(embedding_dim)
        
        # 3. Dimension Calculation
        self.total_input_dim = (len(feature_map['sparse']) * embedding_dim) + \
                               embedding_dim + \
                               feature_map['dense']
        
        # 4. DCN Layer
        self.cross_net = CrossNetV2(self.total_input_dim, num_layers=3)
        
        # 5. Stability Layers
        self.input_norm = nn.LayerNorm(self.total_input_dim)
        self.cross_norm = nn.LayerNorm(self.total_input_dim) # 新增：稳定交叉后的数值
        
        # 6. MMoE Layer
        self.mmoe = MMoELayer(
            input_dim=self.total_input_dim,
            num_experts=num_experts,
            expert_hidden_units=[256, 128],
            num_tasks=len(tasks)
        )
        
        # 6. Task Heads
        self.heads = nn.ModuleDict()
        for task in tasks:
            self.heads[task] = nn.Linear(128, 1)

    def forward(self, inputs):
        """
        核心修复：自动处理 Listwise (3D) 和 Pointwise (2D) 输入。
        """
        # 1. 识别并展平维度 (B, K, ...) -> (B*K, ...)
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

        # A. Embeddings (B*K, E)
        sparse_embs = [self.embeddings[feat](flat_inputs[feat]) for feat in self.feature_map['sparse']]
        
        # B. DIN (B*K, E)
        target_item_emb = self.embeddings['movieId'](flat_inputs['movieId']).unsqueeze(1) # (B*K, 1, E)
        history_seq_emb = self.embeddings['movieId'](flat_inputs['seq_history']) # (B*K, T, E)
        seq_mask = (flat_inputs['seq_history'] > 0).float()
        interest_emb = self.din_attention(target_item_emb, history_seq_emb, seq_mask)
        
        # C. Concatenate (B*K, D_total)
        all_features = torch.cat(sparse_embs + [interest_emb, flat_inputs['dense_feats']], dim=-1)
        
        # 稳定性修正：层归一化
        all_features = self.input_norm(all_features)
        
        # D. DCN Cross
        cross_out = self.cross_net(all_features)
        cross_out = self.cross_norm(cross_out) # 应用归一化
        
        # E. MMoE Experts
        task_specific_features = self.mmoe(cross_out)
        
        # F. Multi-Heads
        outputs = {}
        for i, task in enumerate(self.tasks):
            logits = self.heads[task](task_specific_features[i]).squeeze(-1)
            # 如果是 Listwise，在这里不进行 reshape，由训练脚本根据需要 reshape
            outputs[task] = logits
            
        return outputs
