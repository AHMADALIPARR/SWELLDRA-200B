import torch.nn as nn
from ..modeling.backbone import SwiGLU
class GraphExpert(nn.Module):
    def __init__(self, d_model=8192, hidden_dim=28672):
        super().__init__()
        self.ffn = SwiGLU(d_model, hidden_dim)
        self.relation_proj = nn.Linear(d_model, d_model, bias=False)
        self.expert_type = "graph"
    def forward(self, x, relation_emb=None):
        if relation_emb is not None:
            x = x + self.relation_proj(relation_emb)
        return self.ffn(x)
