import argparse, time, math, json, torch
from pathlib import Path
from transformers import AutoTokenizer, GPT2Config, AutoModelForCausalLM

from spectral_attention import convert_gpt2lm_to_spectral
from spectral_attention.utils import resolve_device, parse_amp_dtype


def set_seed(s=1234):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.inference_mode()
def measure_throughput(model, x, iters=50, warmup=10):
    model.eval()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    for _ in range(warmup): _ = model(input_ids=x, attention_mask=(x != pad_id))
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(input_ids=x, attention_mask=(x != pad_id))
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = x.numel()
    return (toks * iters) / dt, (dt / iters) * 1000.0, (torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.0


def wikitext_ids(tokenizer, dataset="wikitext-2-raw-v1", split="train"):
    """Return a long token stream for training.

    Tries to load Hugging Face wikitext; if unavailable, falls back to synthetic tokens.
    """
    try:
        from datasets import load_dataset  # lazy import to avoid hard dependency
        ds = load_dataset("wikitext", dataset)[split]["text"]
        text = "\n\n".join(ds)
        enc = tokenizer(text, return_tensors=None, truncation=False, padding=False)["input_ids"]
        return torch.tensor(enc, dtype=torch.long)
    except Exception as e:
        print(f"[data] datasets not available or load failed; using synthetic tokens. reason={e}")
        vocab_size = getattr(tokenizer, "vocab_size", 50257)
        # 2M random tokens as a fallback corpus
        return torch.randint(0, vocab_size, (2_000_000,), dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model", default="gpt2", help="base model to load embeddings/MLP/LayerNorm from")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1, help="Weight decay for non-spectral params")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="bf16")
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--use_dct", action="store_true")
    ap.add_argument("--token_gate", action="store_true")
    ap.add_argument("--logdir", type=str, default="experiments/runs/compare")
    ap.add_argument("--log_every", type=int, default=50)
    # Distillation and freezing options
    ap.add_argument("--kd", action="store_true", help="Enable knowledge distillation from a vanilla teacher")
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="Weight of KD loss vs CE loss")
    ap.add_argument("--kd_tau", type=float, default=2.0, help="Distillation temperature")
    ap.add_argument("--teacher_model", type=str, default=None, help="HF model name for teacher (defaults to --hf_model)")
    ap.add_argument("--freeze_non_attn", action="store_true", help="Freeze non-attention weights for a warmup phase")
    # Spectral-specific optimizer tuning
    ap.add_argument("--spectral_lr_scale", type=float, default=2.0, help="Multiply LR for spectral params by this factor")
    ap.add_argument("--spectral_weight_decay", type=float, default=0.01, help="Weight decay for spectral params")
    # Checkpointing
    ap.add_argument("--save_ckpt", type=str, default=None, help="Path to save model state_dict at end of run (.pt)")
    ap.add_argument("--resume_ckpt", type=str, default=None, help="Path to load model state_dict before training (.pt)")
    args = ap.parse_args()

    set_seed(1234)
    device = resolve_device(args.device)
    print(f"[train_lm_hf_spectral] Using device: {device}")
    dtype = parse_amp_dtype(args.dtype) or torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global pad_id
    pad_id = tokenizer.pad_token_id

    # Build spectral GPT-2 LMHead model (load pretrained for non-attn weights)
    cfg = GPT2Config.from_pretrained(args.hf_model)
    model = convert_gpt2lm_to_spectral(cfg, from_pretrained=args.hf_model)
    model = model.to(device).to(dtype)
    model.train()

    # Build optimizer with param groups
    spectral_params = []
    base_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "spectral" in n.lower():
            spectral_params.append(p)
        else:
            base_params.append(p)
    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr, "weight_decay": args.wd})
    if spectral_params:
        param_groups.append({
            "params": spectral_params,
            "lr": args.lr * args.spectral_lr_scale,
            "weight_decay": args.spectral_weight_decay,
        })
    if not param_groups:  # fallback (e.g., all frozen)
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.wd}]
    opt = torch.optim.AdamW(param_groups)
    lossf = torch.nn.CrossEntropyLoss()
    # Optional: resume from checkpoint
    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        if ckpt_path.is_file():
            sd = torch.load(str(ckpt_path), map_location=device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[resume] loaded: {ckpt_path.name} | missing: {len(missing)} | unexpected: {len(unexpected)}")
        else:
            print(f"[resume] WARNING: checkpoint not found at {args.resume_ckpt}")

    # Optional: freeze non-attention weights for a warmup
    if args.freeze_non_attn:
        frozen = unfrozen = 0
        for n, p in model.named_parameters():
            key = n.lower()
            if ("attn" in key) or ("spectral" in key) or ("attention" in key):
                p.requires_grad = True; unfrozen += p.numel()
            else:
                p.requires_grad = False; frozen += p.numel()
        print(f"[freeze] frozen params: {frozen:,} | trainable (attn/spectral): {unfrozen:,}")

    # Optional: initialize teacher for knowledge distillation
    teacher = None
    if args.kd:
        t_name = args.teacher_model or args.hf_model
        teacher = AutoModelForCausalLM.from_pretrained(t_name).to(device)
        teacher.eval()

    # synthetic input for throughput
    x_synth = torch.full((args.batch, args.seq), fill_value=tokenizer.eos_token_id, device=device, dtype=torch.long)
    tok_s, ms, peakMB = measure_throughput(model, x_synth)

    # dataset ids
    ids = wikitext_ids(tokenizer, split="train").to(device)

    def sample_batch():
        N = ids.numel()
        idx = torch.randint(0, max(1, N-args.seq-1), (args.batch,), device=device)
        x = torch.stack([ids[i:i+args.seq] for i in idx])
        y = torch.stack([ids[i+1:i+args.seq+1] for i in idx])
        return x, y

    logs = []
    kind = f"hf_{args.hf_model}_spectral"
    logs.append({"event":"throughput", "task":"lm", "kind": kind, "seq": args.seq,
                 "tokens_per_s": tok_s, "ms_per_it": ms, "peakMB": peakMB})

    for step in range(1, args.steps+1):
        x,y = sample_batch()
        attn = (x != pad_id)
        out = model(input_ids=x, attention_mask=attn)
        logits = out.logits[:, :-1]
        ce = lossf(logits.reshape(-1, logits.size(-1)), y[:, :-1].reshape(-1))

        kd_loss = torch.tensor(0.0, device=device)
        if teacher is not None:
            with torch.no_grad():
                t_out = teacher(input_ids=x, attention_mask=attn)
                t_logits = t_out.logits[:, :-1]
            tau = float(args.kd_tau)
            log_p = torch.log_softmax(logits / tau, dim=-1)
            q = torch.softmax(t_logits / tau, dim=-1)
            kd_loss = torch.nn.functional.kl_div(log_p, q, reduction="batchmean") * (tau * tau)

        alpha = float(args.kd_alpha) if teacher is not None else 0.0
        loss = alpha * kd_loss + (1.0 - alpha) * ce

        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % args.log_every == 0:
            ppl = math.exp(float(ce)) if ce < 20 else float("inf")
            rec = {"event":"train", "kind": kind, "step": step, "loss": float(loss), "ppl": ppl, "ce": float(ce)}
            if teacher is not None:
                rec.update({"kd_loss": float(kd_loss), "alpha": alpha, "tau": float(args.kd_tau)})
            logs.append(rec)

    # quick validation
    with torch.no_grad():
        total, seen = 0.0, 0
        for _ in range(50):
            x,y = sample_batch()
            attn = (x != pad_id)
            out = model(input_ids=x, attention_mask=attn)
            logits = out.logits[:, :-1]
            loss = lossf(logits.reshape(-1, logits.size(-1)), y[:, :-1].reshape(-1))
            total += float(loss); seen += 1
        vavg = total / max(1, seen)
        vppl = math.exp(vavg) if vavg < 20 else float("inf")
        logs.append({"event":"val", "task":"lm", "kind": kind, "seq": args.seq,
                     "val_loss": vavg, "val_ppl": vppl, "loss": vavg, "ppl": vppl})

    # write logs
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.logdir) / f"lm_{kind}_T{args.seq}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"metrics.jsonl", "a", encoding="utf-8") as f:
        for r in logs:
            f.write(json.dumps(r) + "\n")
    print("logs →", str(out_dir))

    # Save checkpoint (state_dict) if requested
    if args.save_ckpt:
        ckpt_path = Path(args.save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(ckpt_path))
        meta = {
            "saved_at": stamp,
            "hf_model": args.hf_model,
            "seq": args.seq,
            "steps": args.steps,
            "kd": args.kd,
            "kd_alpha": args.kd_alpha,
            "kd_tau": args.kd_tau,
            "freeze_non_attn": args.freeze_non_attn,
            "dtype": args.dtype,
        }
        with open(ckpt_path.with_suffix(".json"), "w", encoding="utf-8") as jf:
            json.dump(meta, jf, indent=2)
        print(f"[save] checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
