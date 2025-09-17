import time
import torch

@torch.inference_mode()
def throughput_tokens_per_s(module: torch.nn.Module, x: torch.Tensor, warmup=10, iters=50) -> tuple[float, float]:
    dev = x.device
    module.eval().to(dev)
    # warmup
    for _ in range(warmup):
        _ = module(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = module(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    tokens = x.shape[0] * x.shape[1] * iters
    return tokens / elapsed, (elapsed / iters) * 1000.0

def peak_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)

def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
