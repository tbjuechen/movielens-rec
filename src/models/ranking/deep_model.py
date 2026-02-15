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
        
        # 1. Experts: 一组专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_units[0]),
                nn.ReLU(),
                nn.Linear(expert_hidden_units[0], expert_hidden_units[1])
            ) for _ in range(num_experts)
        ])
        
        # 2. Gates: 每个任务都有一个门控网络
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        # x: (B, input_dim)
        # expert_outputs: (num_experts, B, expert_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        
        final_outputs = []
        for i in range(self.num_tasks):
            # gate_weights: (B, num_experts)
            gate_weights = self.gates[i](x)
            # 加权求和: (B, expert_dim)
            task_out = torch.sum(gate_weights.transpose(0, 1).unsqueeze(-1) * expert_outputs, dim=0)
            final_outputs.append(task_out)
            
        return final_outputs # List of (B, expert_dim)

class UnifiedDeepRanker(nn.Module):
    def __init__(self, 
                 feature_map: dict, 
                 embedding_dim: int = 128,
                 num_experts: int = 4,
                 tasks: list = ['click', 'rating']):
        super().__init__()
        self.feature_map = feature_map
        self.tasks = tasks
        
        # 1. Embedding Layers (扩容至 128)
        self.embeddings = nn.ModuleDict()
        for feat, size in feature_map['sparse'].items():
            self.embeddings[feat] = nn.Embedding(size + 1, embedding_dim, padding_idx=0)
            
        # 2. Sequence Layer (DIN)
        self.din_attention = AttentionSequencePooling(embedding_dim)
        
        # 3. 维度计算
        self.total_input_dim = (len(feature_map['sparse']) * embedding_dim) + \
                               embedding_dim + \
                               feature_map['dense']
        
        # 4. DCN Layer (高阶交叉)
        self.cross_net = CrossNetV2(self.total_input_dim, num_layers=3)
        
        # 5. MMoE Layer (专家系统)
        self.mmoe = MMoELayer(
            input_dim=self.total_input_dim,
            num_experts=num_experts,
            expert_hidden_units=[256, 128],
            num_tasks=len(tasks)
        )
        
        # 6. Task Heads
        self.heads = nn.ModuleDict()
        for i, task in enumerate(tasks):
            # 每个任务基于 MMoE 吐出的 expert 融合向量进行最终预测
            self.heads[task] = nn.Linear(128, 1)

    def forward(self, inputs):
        # A. Embeddings
        sparse_embs = [self.embeddings[feat](inputs[feat]) for feat in self.feature_map['sparse']]
        
        # B. DIN
        target_item_emb = self.embeddings['movieId'](inputs['movieId']).unsqueeze(1)
        history_seq_emb = self.embeddings['movieId'](inputs['seq_history'])
        seq_mask = (inputs['seq_history'] > 0).float()
        interest_emb = self.din_attention(target_item_emb, history_seq_emb, seq_mask)
        
        # C. Concatenate
        all_features = torch.cat(sparse_embs + [interest_emb, inputs['dense_feats']], dim=-1)
        # Reshape if input was 3D (Listwise batching)
        if all_features.dim() == 3:
            B, K, D = all_features.shape
            all_features = all_features.view(B*K, D)
        else:
            B, K = all_features.size(0), 1
        
        # D. DCN Cross
        cross_out = self.cross_net(all_features)
        
        # E. MMoE Experts
        task_specific_features = self.mmoe(cross_out)
        
        # F. Multi-Heads
        outputs = {}
        for i, task in enumerate(self.tasks):
            logits = self.heads[task](task_specific_features[i]).squeeze(-1)
            # 如果是 Listwise 训练，我们在训练脚本里处理 reshape
            outputs[task] = logits
            
        return outputs
