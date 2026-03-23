import torch
import torch.nn as nn


class CrossNetV2(nn.Module):
    """DCN-V2 Cross Network: explicit high-order feature interactions.

    x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
    """

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(num_layers)]
        )

    def forward(self, x0):
        x = x0
        for linear in self.linears:
            x = x0 * linear(x) + x
        return x


class DeepNetwork(nn.Module):
    """MLP with LayerNorm, ReLU, Dropout."""

    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x):
        return self.mlp(x)


class MMoE(nn.Module):
    """Multi-gate Mixture-of-Experts for multi-task learning."""

    def __init__(self, input_dim, num_experts, expert_dim, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, expert_dim),
            )
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts) for _ in range(num_tasks)
        ])

    def forward(self, x):
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, E, D)
        task_outputs = []
        for gate in self.gates:
            gate_w = torch.softmax(gate(x), dim=-1).unsqueeze(-1)  # (B, E, 1)
            task_out = (expert_outs * gate_w).sum(dim=1)  # (B, D)
            task_outputs.append(task_out)
        return task_outputs


class TaskTower(nn.Module):
    """Task-specific prediction tower."""

    def __init__(self, input_dim, hidden_dims, use_sigmoid=False):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        out = self.mlp(x).squeeze(-1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out
