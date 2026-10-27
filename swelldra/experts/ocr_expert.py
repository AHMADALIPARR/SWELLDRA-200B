import torch.nn as nn
from ..modeling.backbone import SwiGLU
class OCRExpert(nn.Module):
    def __init__(self, d_model=8192, hidden_dim=28672, vision_dim=1024):
        super().__init__()
        self.ffn = SwiGLU(d_model, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, d_model, bias=False)
        self.layout_bias = nn.Parameter(nn.Parameter(torch.zeros(d_model)))
        self.expert_type = "ocr"
    def forward(self, x, vision_tokens=None, layout_info=None):
        if vision_tokens is not None:
            v = self.vision_proj(vision_tokens)
            x = x + v
        return self.ffn(x)
