import torch.nn as nn
from ..modeling.backbone import SwiGLU
class LanguageExpert(nn.Module):
    def __init__(self, d_model=8192, hidden_dim=28672):
        super().__init__()
        self.ffn = SwiGLU(d_model, hidden_dim)
        self.expert_type = "language"
    def forward(self, x):
        return self.ffn(x)
