import torch

def swell_embedding(old_embed, new_vocab_size, new_d_model):
    old_vocab, old_dim = old_embed.shape
    new_embed = torch.zeros(new_vocab_size, new_d_model)
    new_embed[:old_vocab, :old_dim] = old_embed
    if new_d_model > old_dim:
        repeat_times = new_d_model // old_dim
        remainder = new_d_model % old_dim
        tiled = old_embed.repeat(1, repeat_times)
        if remainder:
            tiled = torch.cat([tiled, old_embed[:, :remainder]], dim=1)
        noise = torch.randn_like(tiled) * 0.01
        new_embed[:old_vocab, :] = tiled + noise
    if new_vocab_size > old_vocab:
        mean = old_embed.mean(dim=0)
        mean_expanded = torch.zeros(new_d_model)
        mean_expanded[:old_dim] = mean
        for i in range(old_vocab, new_vocab_size):
            new_embed[i] = mean_expanded + torch.randn(new_d_model) * 0.02
    return new_embed

class SwellingScheduler:
    LADDER = [
        ("1m", "configs/moe_1m.yaml"),
        ("100m", "configs/moe_100m.yaml"),
        ("1b", "configs/moe_1b.yaml"),
        ("7b", "configs/moe_7b.yaml"),
        ("45b", "configs/moe_45b.yaml"),
        ("200b", "configs/moe_200b.yaml"),
    ]
    def should_advance(self, metrics):
        thresholds = {"ocr_acc": 0.85, "sql_exec_acc": 0.70, "graph_acc": 0.80, "logic_verif": 0.95, "router_max_load": 2.0}
        for k, thresh in thresholds.items():
            if k == "router_max_load":
                if metrics.get(k, 999) > thresh:
                    return False
            else:
                if metrics.get(k, 0) < thresh:
                    return False
        return True
