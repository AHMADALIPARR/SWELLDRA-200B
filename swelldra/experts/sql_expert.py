import torch.nn as nn
from ..modeling.backbone import SwiGLU
class SQLExpert(nn.Module):
    def __init__(self, d_model=8192, hidden_dim=28672):
        super().__init__()
        self.ffn = SwiGLU(d_model, hidden_dim)
        self.schema_proj = nn.Linear(d_model, d_model, bias=False)
        self.expert_type = "sql"
    def forward(self, x, schema_emb=None, feedback_emb=None):
        if schema_emb is not None:
            x = x + self.schema_proj(schema_emb)
        return self.ffn(x)
