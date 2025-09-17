# scripts/bench_spectral_attention.py
import time
import argparse
import json
from pathlib import Path
import torch

from spectral_attention import SpectralAttention
from spectral_attention.vanilla_blocks import VanillaEncoder

# ---- Global perf toggles ----
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    # Allow TF32 for fast matmuls/convs on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def parse_device(device_flag: str) -> torch.device:
    device_flag = (device_flag or "auto").lower()
    if device_flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_flag == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA not available")
        return torch.device("cuda")
    if device_flag == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device option: {device_flag}")


def bench(block: torch.nn.Module, x: torch.Tensor, warmup: int = 10, iters: int = 50):
    block.eval()

    # Safety: everything must be on the same device
    dev = x.device
    assert all(p.device == dev for p in block.parameters()), "Params not on x.device"
    # Some modules may have no buffers; guard iterator
    for b in block.buffers():
        assert b.device == dev, "A buffer is not on x.device"
        break

    with torch.inference_mode():
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # Warmup
        for _ in range(warmup):
            _ = block(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        # Timed loop
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = block(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    elapsed = t1 - t0
    tokens = x.shape[0] * x.shape[1] * iters
    tok_per_s = tokens / elapsed
    ms_per_iter = (elapsed / iters) * 1000.0
    return tok_per_s, ms_per_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", type=str, choices=["spectral", "vanilla"], default="spectral")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--dmodel", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "gpu"],
                    default="gpu" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--logdir", type=str, default=None, help="If set, write metrics.jsonl here")
    #ap.add_argument("--compile", action="store_true", help="Use torch.compile if available")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--compile", action="store_true", help="Use torch.compile")
    ap.add_argument("--backend", type=str, default="inductor",
                choices=["inductor", "eager", "aot_eager"],
                help="torch.compile backend")

    args = ap.parse_args()

    B, T, D, H = args.batch, args.seq, args.dmodel, args.heads
    device = parse_device(args.device)

    # Input on chosen device
    x = torch.randn(B, T, D, device=device)

    print(f"Device={args.device}  (tensor on {device.type})  B={B}  T={T}  d_model={D}  heads={H}  depth={args.depth}  kind={args.kind}")
    if args.debug and device.type == "cuda":
        print("[debug] capability:", torch.cuda.get_device_capability(),
              "| allow_tf32:", torch.backends.cuda.matmul.allow_tf32,
              "| cudnn_tf32:", torch.backends.cudnn.allow_tf32)

    # Build model stack by kind
    if args.kind == "vanilla":
        model = VanillaEncoder(depth=args.depth, d_model=D, n_heads=H).to(device)
        label = "vanilla"
    else:
        # Default spectral config: DCT path without token gate for stability
        blocks = [
            SpectralAttention(D, H, use_dct=True, token_gate=False, device=args.device)
            for _ in range(args.depth)
        ]
        model = torch.nn.Sequential(*blocks, torch.nn.LayerNorm(D)).to(device)
        label = "spectral_dct"

    # Optional compile (guard: only when requested)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False, backend=args.backend)
        except Exception as e:
            print("[warn] torch.compile failed; continuing in eager:", e)

    # Benchmark
    tps, ms = bench(model, x, warmup=args.warmup, iters=args.iters)
    peakMB = 0.0
    if device.type == "cuda":
        peakMB = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"{label:>12s}  tokens/s: {tps:,.0f}   ms/iter: {ms:.2f}   peakMB: {peakMB:.1f}")

    # Optional logging
    if args.logdir:
        out_dir = Path(args.logdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outp = out_dir / "metrics.jsonl"
        rec = {
            "event": "throughput",
            "task": "attention_bench",
            "kind": args.kind,
            "seq": T,
            "batch": B,
            "dmodel": D,
            "heads": H,
            "depth": args.depth,
            "device": args.device,
            "tokens_per_s": float(tps),
            "ms_per_it": float(ms),
            "peakMB": float(peakMB),
        }
        with open(outp, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
