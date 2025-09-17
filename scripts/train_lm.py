# scripts/train_lm.py
import os, json, time, math, argparse
from pathlib import Path
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

from spectral_attention.models import SpectralLM, VanillaLM
from spectral_attention.utils import resolve_device
from spectral_attention import SpectralAttention  # for freq plot hook

def set_seed(s=1337):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_device(which):
    return resolve_device(which)

def make_model(kind, vocab_size, d_model, n_heads, depth, max_len, dropout, use_dct):
    if kind == "spectral":
        return SpectralLM(vocab_size, d_model, n_heads, depth, max_len, dropout, use_dct)
    elif kind == "vanilla":
        return VanillaLM(vocab_size, d_model, n_heads, depth, max_len, dropout)
    raise ValueError(kind)

def save_jsonl(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

@torch.inference_mode()
def measure_throughput(model, batch, device, iters=30, warmup=10):
    model.eval().to(device)
    x = batch.to(device)
    for _ in range(warmup): _ = model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): _ = model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = x.numel()  # B*T tokens
    return (toks * iters) / dt, (dt/iters)*1000.0

def spectral_attn_from(model):
    # find first SpectralAttention inside the model, for plotting
    for m in model.modules():
        if isinstance(m, SpectralAttention):
            return m
    return None

def plot_freq(attn, out_png):
    import matplotlib.pyplot as plt
    g = torch.exp(attn.log_gain).mean(0).detach().cpu().numpy()
    p = attn.phase.mean(0).detach().cpu().numpy()
    fig, ax = plt.subplots(2,1, figsize=(8,6), tight_layout=True)
    ax[0].plot(g); ax[0].set_title("Mean Gain"); ax[0].set_xlabel("bin"); ax[0].set_ylabel("gain")
    ax[1].plot(p); ax[1].set_title("Mean Phase"); ax[1].set_xlabel("bin"); ax[1].set_ylabel("rad")
    fig.savefig(out_png, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["spectral","vanilla"], required=True)
    ap.add_argument("--dataset", default="wikitext-2-raw-v1")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--dmodel", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_dct", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--mixed_precision", choices=["bf16","fp16","none"], default="bf16")
    ap.add_argument("--outdir", default="experiments/runs/lm")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--plot_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[train_lm] Using device: {device}")

    # data
    ds = load_dataset("wikitext", args.dataset)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tok.model_max_length = args.seq
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

    def encode(split):
        text = "\n\n".join(ds[split]["text"])
        ids = tok(text, return_tensors=None, padding=False, truncation=False)["input_ids"]
        import numpy as np
        return torch.tensor(ids, dtype=torch.long)
    train_ids = encode("train")
    val_ids   = encode("validation")
    vocab_size = tok.vocab_size

    def sample_batch(ids, B, T):
        N = ids.numel()
        ix = torch.randint(0, max(1, N - T - 1), (B,))
        x = torch.stack([ids[i:i+T]     for i in ix]).to(device)
        y = torch.stack([ids[i+1:i+T+1] for i in ix]).to(device)
        return x, y

    model = make_model(args.kind, vocab_size, args.dmodel, args.heads, args.depth, args.seq, args.dropout, args.use_dct).to(device)

    if args.compile:
        model = torch.compile(model, backend="inductor")

    scaler = None
    amp_dtype = None
    if args.mixed_precision != "none":
        if args.mixed_precision == "bf16":
            amp_dtype = torch.bfloat16
        elif args.mixed_precision == "fp16":
            amp_dtype = torch.float16

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.CrossEntropyLoss()

    # resume
    start_step = 0
    best_val = float("inf")
    outdir = Path(args.outdir) / time.strftime(f"{args.kind}_T{args.seq}_%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.jsonl"

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", best_val)

    # initial throughput
    tps, ms = measure_throughput(model, torch.randint(0, vocab_size, (args.batch, args.seq)), device)
    save_jsonl(metrics_path, {"event":"throughput","kind":args.kind,"seq":args.seq,"tokens_per_s":tps,"ms_per_it":ms})

    def evaluate(split_ids, max_batches=100):
        model.eval()
        total_loss, seen = 0.0, 0
        with torch.no_grad():
            for _ in range(max_batches):
                x, y = sample_batch(split_ids, args.batch, args.seq)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    logits = model(x)[:, :-1]
                    loss = lossf(logits.reshape(-1, vocab_size), y[:, :-1].reshape(-1))
                total_loss += float(loss)
                seen += 1
        model.train()
        avg = total_loss / max(1, seen)
        ppl = math.exp(avg) if avg < 20 else float("inf")
        return avg, ppl

    # training
    model.train()
    for step in range(start_step+1, args.steps+1):
        x, y = sample_batch(train_ids, args.batch, args.seq)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logits = model(x)[:, :-1]                # causal loss
            loss = lossf(logits.reshape(-1, vocab_size), y[:, :-1].reshape(-1)) / args.grad_accum

        loss.backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            # loss was scaled by grad_accum earlier; log full-step loss and train ppl
            train_loss = float(loss) * args.grad_accum
            train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")
            save_jsonl(metrics_path, {"event":"train","step":step,
                                      "loss":train_loss,
                                      "ppl":train_ppl})

        if step % args.save_every == 0 or step == args.steps:
            # val
            vloss, vppl = evaluate(val_ids, max_batches=50)
            # Log both legacy (val_loss/val_ppl) and unified (loss/ppl) keys for downstream tools
            save_jsonl(metrics_path, {"event":"val","step":step,
                                      "val_loss":vloss, "val_ppl":vppl,
                                      "loss":vloss,   "ppl":vppl})
            # save last
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best_val": best_val}, outdir / f"ckpt_step{step}.pt")
            # save best
            if vloss < best_val:
                best_val = vloss
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "best_val": best_val}, outdir / f"ckpt_best.pt")

        if args.kind == "spectral" and args.plot_every > 0 and step % args.plot_every == 0:
            sa = spectral_attn_from(model)
            if sa is not None:
                png = outdir / f"freq_step{step}.png"
                plot_freq(sa, png)

    print("done ->", outdir)

if __name__ == "__main__":
    main()
