import argparse, time, math, json, torch, os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.inference_mode()
def measure_throughput(model, x, attn, iters=50, warmup=10):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(input_ids=x, attention_mask=attn)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(input_ids=x, attention_mask=attn)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = x.numel()
    return (toks * iters) / dt, (dt / iters) * 1000.0

@torch.inference_mode()
def eval_perplexity(model, tokenizer, seq, batch, max_batches=100, dataset="wikitext-2-raw-v1", split="validation"):
    ds = load_dataset("wikitext", dataset)[split]["text"]
    text = "\n\n".join(ds)
    enc = tokenizer(text, return_tensors=None, truncation=False, padding=False)["input_ids"]
    ids = torch.tensor(enc, dtype=torch.long, device=device)
    lossf = torch.nn.CrossEntropyLoss()
    total, seen = 0.0, 0
    for _ in range(max_batches):
        N = ids.numel()
        idx = torch.randint(0, max(1, N - seq - 1), (batch,))
        x = torch.stack([ids[i:i+seq] for i in idx])
        y = torch.stack([ids[i+1:i+seq+1] for i in idx])
        # No padding in this slicing scenario; full attention
        attn = torch.ones_like(x, dtype=torch.bool, device=device)
        out = model(input_ids=x, attention_mask=attn)
        logits = out.logits[:, :-1]
        loss = lossf(logits.reshape(-1, logits.size(-1)), y[:, :-1].reshape(-1))
        total += float(loss); seen += 1
    avg = total / max(1, seen)
    ppl = math.exp(avg) if avg < 20 else float("inf")
    return avg, ppl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model", default="gpt2", help="e.g. gpt2, gpt2-medium, meta-llama/Llama-3.2-1B")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="bf16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--ppl", action="store_true", help="also compute perplexity on wikitext-2-raw-v1 val")
    ap.add_argument("--logdir", type=str, default=None, help="If set, write metrics.jsonl under this directory for notebook aggregation")
    args = ap.parse_args()

    global device, pad_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=dtype, device_map=None)
    model = model.to(device)
    # Try to compile only if Triton is available and CUDA is present; otherwise safely skip
    if args.compile and torch.cuda.is_available():
        try:
            import triton  # noqa: F401
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True  # fall back to eager on any compile failure
            model = torch.compile(model, backend="inductor", mode="max-autotune")
        except Exception as e:
            print("[warn] compile unavailable or failed (falling back to eager):", e)
    else:
        if args.compile:
            print("[warn] --compile requested but CUDA or Triton not available; running in eager mode.")

    # Determine model's maximum supported sequence length and cap if needed
    cfg = getattr(model, "config", None)
    max_pos = None
    for k in ("n_positions", "max_position_embeddings", "max_positions"):
        if cfg is not None and hasattr(cfg, k):
            val = getattr(cfg, k)
            if isinstance(val, int):
                max_pos = val
                break
    effective_seq = args.seq if (max_pos is None) else min(args.seq, max_pos)
    if effective_seq < args.seq:
        print(f"[warn] Requested seq={args.seq} exceeds model max={max_pos}; using seq={effective_seq} instead.")

    # synthetic input for fair speed test (no decoding, just forward)
    vocab_size = getattr(model.config, "vocab_size", tokenizer.vocab_size)
    x = torch.randint(low=0, high=vocab_size, size=(args.batch, effective_seq), device=device, dtype=torch.long)
    attn_mask = torch.ones_like(x, dtype=torch.bool, device=device)

    tok_s, ms = measure_throughput(model, x, attn_mask, iters=args.iters, warmup=args.warmup)
    peakMB = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    thr_rec = {"event":"throughput", "task":"lm", "kind": f"hf_{args.hf_model}", "seq": effective_seq,
               "requested_seq": args.seq, "hf_model": args.hf_model, "tokens_per_s": tok_s, "ms_per_it": ms, "peakMB": peakMB, "dtype": args.dtype}
    print(thr_rec)

    val_rec = None
    if args.ppl:
        vloss, vppl = eval_perplexity(model, tokenizer, effective_seq, args.batch)
        val_rec = {"event":"val", "task":"lm", "kind": f"hf_{args.hf_model}", "seq": effective_seq,
                   "requested_seq": args.seq,
                   "val_loss": vloss, "val_ppl": vppl, "loss": vloss, "ppl": vppl}
        print(val_rec)

    # Optional logging to metrics.jsonl for aggregation
    if args.logdir:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"lm_hf_{args.hf_model}_T{args.seq}_{stamp}"
        out_dir = Path(args.logdir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        outp = out_dir / "metrics.jsonl"
        with open(outp, "a", encoding="utf-8") as f:
            f.write(json.dumps(thr_rec) + "\n")
            if val_rec is not None:
                f.write(json.dumps(val_rec) + "\n")
        print("logs →", str(out_dir))
