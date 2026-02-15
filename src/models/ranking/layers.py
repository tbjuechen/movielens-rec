import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossNetV2(nn.Module):
    """
    DCN-V2 核心模块：显式高阶特征交叉
    公式：x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
    """
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        # 这里的实现是 DCN-Mix 的简化版，全矩阵相乘
        self.kernels = nn.ParameterList([nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(num_layers)])
        self.bias = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(num_layers)])
        
        for w in self.kernels:
            nn.init.xavier_normal_(w)
        for b in self.bias:
            nn.init.zeros_(b)

    def forward(self, x0):
        # x0 shape: (batch_size, input_dim)
        xl = x0
        for i in range(self.num_layers):
            # x_l * W_l + b_l
            linear = torch.mm(xl, self.kernels[i]) + self.bias[i]
            # x_0 * linear + x_l
            xl = x0 * linear + xl
        return xl

class AttentionSequencePooling(nn.Module):
    """
    DIN 核心模块：基于 Target 的序列注意力机制
    """
    def __init__(self, embed_dim: int, hidden_units=[80, 40]):
        super().__init__()
        # Attention MLP: 输入是 (Query, Key, Query-Key, Query*Key)
        self.mlp = nn.Sequential()
        input_dim = embed_dim * 4
        for i, unit in enumerate(hidden_units):
            self.mlp.add_module(f'fc_{i}', nn.Linear(input_dim, unit))
            self.mlp.add_module(f'act_{i}', nn.PReLU()) # DIN 推荐 PReLU
            input_dim = unit
        self.mlp.add_module('out', nn.Linear(input_dim, 1))

    def forward(self, query, keys, keys_mask=None):
        """
        query: 当前候选物品 (B, 1, E)
        keys: 用户历史行为序列 (B, T, E)
        keys_mask: 序列掩码 (B, T), 1为真实数据, 0为Padding
        """
        # query: (B, 1, E) -> (B, T, E)
        T = keys.size(1)
        query_expanded = query.expand(-1, T, -1)
        
        # DIN 经典组合: [Q, K, Q-K, Q*K]
        interaction = torch.cat([
            query_expanded,
            keys,
            query_expanded - keys,
            query_expanded * keys
        ], dim=-1) # (B, T, 4E)
        
        # 计算 scores
        scores = self.mlp(interaction).squeeze(-1) # (B, T)
        
        if keys_mask is not None:
            scores = scores.masked_fill(keys_mask == 0, -1e9)
            
        scores = F.softmax(scores, dim=1) # (B, T)
        
        # 加权求和
        output = torch.bmm(scores.unsqueeze(1), keys).squeeze(1) # (B, E)
        return output
