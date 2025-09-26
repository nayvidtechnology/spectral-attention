#!/usr/bin/env python3
"""
Benchmark script comparing SpectralAttention with and without FlashAttention.
Measures memory usage and throughput for different sequence lengths.
"""

import argparse
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple

# Add src to path for local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectral_attention import SpectralAttention


def get_memory_stats() -> Dict[str, float]:
    """Get current memory statistics."""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "peak_mb": torch.cuda.max_memory_allocated() / (1024**2)
        }
    else:
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}


def benchmark_attention(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    warmup: int = 5,
    iters: int = 20,
    device: str = "cuda"
) -> Tuple[float, float, Dict[str, float]]:
    """
    Benchmark a SpectralAttention model.
    
    Returns:
        tokens_per_sec: Throughput in tokens/second
        ms_per_iter: Time per iteration in milliseconds  
        memory_stats: Dictionary with memory usage statistics
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    
    # Clear memory stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model, device=device_obj, dtype=torch.float16)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    # Sync before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time = end_time - start_time
    ms_per_iter = (total_time / iters) * 1000
    total_tokens = batch_size * seq_len * iters
    tokens_per_sec = total_tokens / total_time
    
    memory_stats = get_memory_stats()
    
    return tokens_per_sec, ms_per_iter, memory_stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark SpectralAttention with/without FlashAttention")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 2048, 4096, 8192], 
                       help="Sequence lengths to test")
    parser.add_argument("--dmodel", type=int, default=512, help="Model dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--use_dct", action="store_true", help="Use DCT instead of rFFT")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Running benchmarks on device: {device}")
    print(f"Sequence lengths: {args.seq}")
    print(f"Batch size: {args.batch}, d_model: {args.dmodel}, heads: {args.heads}")
    print(f"Transform: {'DCT' if args.use_dct else 'rFFT'}")
    print()
    
    results = []
    
    for seq_len in args.seq:
        print(f"Benchmarking sequence length: {seq_len}")
        
        # Test without FlashAttention
        model_no_flash = SpectralAttention(
            d_model=args.dmodel,
            n_heads=args.heads,
            use_dct=args.use_dct,
            use_flash=False
        )
        
        try:
            tps_no_flash, ms_no_flash, mem_no_flash = benchmark_attention(
                model_no_flash, args.batch, seq_len, args.dmodel,
                args.warmup, args.iters, args.device
            )
            
            print(f"  No Flash:  {tps_no_flash:>8,.0f} tok/s  "
                  f"{ms_no_flash:>6.2f} ms/iter  "
                  f"Peak: {mem_no_flash['peak_mb']:>6.1f} MB")
        except Exception as e:
            print(f"  No Flash: FAILED - {e}")
            tps_no_flash = ms_no_flash = 0.0
            mem_no_flash = {"peak_mb": 0.0, "allocated_mb": 0.0, "reserved_mb": 0.0}
        
        # Test with FlashAttention
        model_flash = SpectralAttention(
            d_model=args.dmodel,
            n_heads=args.heads,
            use_dct=args.use_dct,
            use_flash=True
        )
        
        try:
            tps_flash, ms_flash, mem_flash = benchmark_attention(
                model_flash, args.batch, seq_len, args.dmodel,
                args.warmup, args.iters, args.device
            )
            
            print(f"  Flash:     {tps_flash:>8,.0f} tok/s  "
                  f"{ms_flash:>6.2f} ms/iter  "
                  f"Peak: {mem_flash['peak_mb']:>6.1f} MB")
            
            # Calculate speedup and memory savings
            if tps_no_flash > 0:
                speedup = tps_flash / tps_no_flash
                memory_ratio = mem_flash['peak_mb'] / mem_no_flash['peak_mb'] if mem_no_flash['peak_mb'] > 0 else 1.0
                print(f"  Speedup: {speedup:.2f}x  Memory ratio: {memory_ratio:.2f}x")
            
        except Exception as e:
            print(f"  Flash: FAILED - {e}")
            tps_flash = ms_flash = 0.0
            mem_flash = {"peak_mb": 0.0, "allocated_mb": 0.0, "reserved_mb": 0.0}
        
        # Record results
        result = {
            "seq_len": seq_len,
            "batch_size": args.batch,
            "d_model": args.dmodel,
            "n_heads": args.heads,
            "use_dct": args.use_dct,
            "device": str(device),
            "no_flash": {
                "tokens_per_sec": float(tps_no_flash),
                "ms_per_iter": float(ms_no_flash),
                "memory_stats": mem_no_flash
            },
            "flash": {
                "tokens_per_sec": float(tps_flash),
                "ms_per_iter": float(ms_flash),
                "memory_stats": mem_flash
            }
        }
        results.append(result)
        print()
    
    # Save results if requested
    if args.logdir:
        log_dir = Path(args.logdir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_file = log_dir / "flashspectra_benchmark.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary as JSONL for analysis
        jsonl_file = log_dir / "flashspectra_metrics.jsonl"
        with open(jsonl_file, 'w') as f:
            for result in results:
                # Write no-flash result
                no_flash_record = {
                    "event": "flashspectra_benchmark",
                    "variant": "no_flash",
                    "seq_len": result["seq_len"],
                    "batch_size": result["batch_size"],
                    "d_model": result["d_model"],
                    "n_heads": result["n_heads"],
                    "use_dct": result["use_dct"],
                    "device": result["device"],
                    "tokens_per_sec": result["no_flash"]["tokens_per_sec"],
                    "ms_per_iter": result["no_flash"]["ms_per_iter"],
                    "peak_memory_mb": result["no_flash"]["memory_stats"]["peak_mb"]
                }
                f.write(json.dumps(no_flash_record) + '\n')
                
                # Write flash result
                flash_record = {
                    "event": "flashspectra_benchmark",
                    "variant": "flash",
                    "seq_len": result["seq_len"],
                    "batch_size": result["batch_size"],
                    "d_model": result["d_model"],
                    "n_heads": result["n_heads"],
                    "use_dct": result["use_dct"],
                    "device": result["device"],
                    "tokens_per_sec": result["flash"]["tokens_per_sec"],
                    "ms_per_iter": result["flash"]["ms_per_iter"],
                    "peak_memory_mb": result["flash"]["memory_stats"]["peak_mb"]
                }
                f.write(json.dumps(flash_record) + '\n')
        
        print(f"Results saved to {results_file} and {jsonl_file}")


if __name__ == "__main__":
    main()