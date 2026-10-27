import torch
import torch.nn as nn
from .backbone import TransformerBlock, RMSNorm, GroupedQueryAttention, SwiGLU
from .moe_layer import MoELayer
import yaml

class SwelldraMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        self.d_model = model_cfg['d_model']
        self.n_layers = model_cfg['n_layers']
        self.vocab_size = model_cfg['vocab_size']
        self.max_seq_len = model_cfg.get('max_seq_len', 8192)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        moe_cfg = model_cfg.get('moe', {})
        self.num_experts = moe_cfg.get('num_experts', 4)
        self.top_k = moe_cfg.get('num_experts_per_token', 2)
        self.expert_types = moe_cfg.get('expert_types', {"language": self.num_experts})
        hidden_dim = model_cfg.get('hidden_dim', int(self.d_model * 3.5))
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if i < 4:
                layer = nn.Module()
                layer.attn_norm = RMSNorm(self.d_model)
                layer.attn = GroupedQueryAttention(self.d_model, model_cfg['n_heads'], model_cfg['n_kv_heads'], self.max_seq_len)
                layer.ffn_norm = RMSNorm(self.d_model)
                layer.ffn = SwiGLU(self.d_model, hidden_dim)
                layer.is_moe = False
                def make_dense_forward(block):
                    def fwd(x, mask=None):
                        residual = x
                        x_attn = block.attn(block.attn_norm(x), mask)
                        x = residual + x_attn
                        residual = x
                        x = residual + block.ffn(block.ffn_norm(x))
                        return x
                    return fwd
                layer.forward = make_dense_forward(layer)
            else:
                layer = nn.Module()
                layer.attn_norm = RMSNorm(self.d_model)
                layer.attn = GroupedQueryAttention(self.d_model, model_cfg['n_heads'], model_cfg['n_kv_heads'], self.max_seq_len)
                layer.moe = MoELayer(self.d_model, hidden_dim, self.num_experts, self.top_k, self.expert_types)
                layer.is_moe = True
                def make_moe_forward(block):
                    def fwd(x, mask=None):
                        residual = x
                        x_attn = block.attn(block.attn_norm(x), mask)
                        x = residual + x_attn
                        x_moe, aux_loss, metrics = block.moe(x)
                        return x_moe, aux_loss, metrics
                    return fwd
                layer.forward = make_moe_forward(layer)
            self.layers.append(layer)
        self.final_norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        total_aux_loss = 0.0
        all_metrics = []
        for layer in self.layers:
            if layer.is_moe:
                residual = x
                x_attn = layer.attn(layer.attn_norm(x), attention_mask)
                x = residual + x_attn
                x, aux_loss, metrics = layer.moe(x)
                total_aux_loss = total_aux_loss + aux_loss
                all_metrics.append(metrics)
            else:
                x = layer(x, attention_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return {"logits": logits, "aux_loss": total_aux_loss, "metrics": all_metrics}
    @classmethod
    def from_config_file(cls, path):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        non_expert = 0
        moe_total = 0
        for name, p in self.named_parameters():
            if "experts" in name:
                moe_total += p.numel()
            else:
                non_expert += p.numel()
        if self.num_experts > 0 and moe_total>0:
            per_expert = moe_total // self.num_experts
            active = non_expert + (per_expert * self.top_k * (self.n_layers - 4))
        else:
            active = total
        return {"total": total, "active_per_token": active, "non_expert": non_expert, "moe_total": moe_total}

if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/moe_1m.yaml"
    model = SwelldraMoE.from_config_file(cfg_path)
    params = model.count_parameters()
    print(f"Total: {params['total']:,} | Active: {params['active_per_token']:,}")
