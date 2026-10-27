import torch
import torch.distributed as dist
def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(dist.get_rank() % torch.cuda.device_count()))
def estimate_memory():
    total_params = 200.2e9
    weight_mem = total_params * 2 / 1e9
    optim_mem = total_params * 4 * 2 / 1e9
    total = weight_mem + optim_mem
    print(f"Weight: {weight_mem:.1f}GB | Optim: {optim_mem:.1f}GB | Total: {total:.1f}GB")
    print(f"Per GPU (32): {total/32:.1f}GB")
