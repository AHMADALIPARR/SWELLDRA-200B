import torch.nn as nn
from ..modeling.backbone import SwiGLU
class LogicExpert(nn.Module):
    def __init__(self, d_model=8192, hidden_dim=28672):
        super().__init__()
        self.ffn = SwiGLU(d_model, hidden_dim)
        self.proof_state_proj = nn.Linear(d_model, d_model, bias=False)
        self.expert_type = "logic"
    def forward(self, x, proof_state=None):
        if proof_state is not None:
            x = x + self.proof_state_proj(proof_state)
        return self.ffn(x)
