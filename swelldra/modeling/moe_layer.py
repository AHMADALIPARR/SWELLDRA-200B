import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import SwiGLU, RMSNorm

class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim, expert_type="language"):
        super().__init__()
        self.expert_type = expert_type
        self.ffn = SwiGLU(d_model, hidden_dim)
        self.type_scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return self.ffn(x) * self.type_scale

class MoELayer(nn.Module):
    def __init__(self, d_model=8192, hidden_dim=28672, num_experts=128, top_k=2, expert_types_config=None):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        if expert_types_config is None:
            expert_types_config = {"language": num_experts}
        self.experts = nn.ModuleList()
        self.expert_type_list = []
        for exp_type, count in expert_types_config.items():
            for _ in range(count):
                self.experts.append(Expert(d_model, hidden_dim, exp_type))
                self.expert_type_list.append(exp_type)
        while len(self.experts) < num_experts:
            self.experts.append(Expert(d_model, hidden_dim, "language"))
            self.expert_type_list.append("language")
        from .router import MoERouter
        self.router = MoERouter(d_model, num_experts, top_k)
        self.norm = RMSNorm(d_model)
    def forward(self, x):
        B, S, D = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(-1, D)
        top_indices, top_scores, aux_loss, metrics = self.router(x_flat)
        N = x_flat.shape[0]
        out_flat = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            mask = (top_indices == expert_idx)
            if not mask.any():
                continue
            token_idx, k_idx = torch.where(mask)
            if len(token_idx) == 0:
                continue
            tokens = x_flat[token_idx]
            expert_out = expert(tokens)
            weights = top_scores[token_idx, k_idx].unsqueeze(-1)
            expert_out = expert_out * weights
            out_flat.index_add_(0, token_idx, expert_out)
        out = out_flat.view(B, S, D)
        out = out + x
        return out, aux_loss, metrics
