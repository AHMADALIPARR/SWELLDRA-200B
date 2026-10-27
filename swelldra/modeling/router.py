import torch
import torch.nn as nn
import torch.nn.functional as F

class MoERouter(nn.Module):
    def __init__(self, d_model=8192, num_experts=128, top_k=2, capacity_factor=1.25, aux_loss_weight=0.01):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_epsilon = 1e-2

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 3:
            x_flat = x.view(-1, x.shape[-1])
        else:
            x_flat = x
        logits = self.gate(x_flat)
        if self.training:
            noise = torch.randn_like(logits) * self.noise_epsilon
            logits = logits + noise
        scores = F.softmax(logits, dim=-1)
        top_scores, top_indices = torch.topk(scores, k=self.top_k, dim=-1)
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)
        mask = F.one_hot(top_indices, num_classes=self.num_experts).float()
        tokens_per_expert = mask.sum(dim=0).sum(dim=0)
        f_i = tokens_per_expert / x_flat.shape[0]
        P_i = scores.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f_i * P_i) * self.aux_loss_weight
        metrics = {
            "aux_loss": aux_loss.detach(),
            "router_entropy": -(scores * torch.log(scores + 1e-9)).sum(dim=-1).mean().detach(),
            "max_expert_load": f_i.max().detach(),
            "min_expert_load": f_i.min().detach(),
            "mean_expert_load": f_i.mean().detach(),
        }
        if x.dim() == 3:
            B, S, _ = orig_shape
            top_indices = top_indices.view(B, S, self.top_k)
            top_scores = top_scores.view(B, S, self.top_k)
        return top_indices, top_scores, aux_loss, metrics
