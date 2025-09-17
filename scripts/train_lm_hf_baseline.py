import argparse, time, math, json, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Simple HF-from-scratch training loop to match our compare/train logging schema

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
    ds = load_dataset("wikitext", dataset)[split]["text"]
    text = "\n\n".join(ds)
    enc = tokenizer(text, return_tensors=None, truncation=False, padding=False)["input_ids"]
    return torch.tensor(enc, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model", default="gpt2", help="e.g. gpt2")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="bf16")
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--logdir", type=str, default="experiments/runs/compare")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    set_seed(1234)
    device = torch.device("cuda" if (args.device!="cpu" and torch.cuda.is_available()) else "cpu")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global pad_id
    pad_id = tokenizer.pad_token_id

    # from-scratch init (no pre-trained weights)
    model = AutoModelForCausalLM.from_config(AutoModelForCausalLM.from_pretrained(args.hf_model).config)
    model = model.to(device).to(dtype)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = torch.nn.CrossEntropyLoss()

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
    logs.append({"event":"throughput", "task":"lm", "kind": f"hf_{args.hf_model}_scratch", "seq": args.seq,
                 "tokens_per_s": tok_s, "ms_per_it": ms, "peakMB": peakMB})

    for step in range(1, args.steps+1):
        x,y = sample_batch()
        attn = (x != pad_id)
        out = model(input_ids=x, attention_mask=attn)
        logits = out.logits[:, :-1]
        loss = lossf(logits.reshape(-1, logits.size(-1)), y[:, :-1].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % args.log_every == 0:
            ppl = math.exp(float(loss)) if loss < 20 else float("inf")
            logs.append({"event":"train", "kind": f"hf_{args.hf_model}_scratch", "step": step, "loss": float(loss), "ppl": ppl})

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
        logs.append({"event":"val", "task":"lm", "kind": f"hf_{args.hf_model}_scratch", "seq": args.seq,
                     "val_loss": vavg, "val_ppl": vppl, "loss": vavg, "ppl": vppl})

    # write logs
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.logdir) / f"lm_hf_{args.hf_model}_scratch_T{args.seq}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"metrics.jsonl", "a", encoding="utf-8") as f:
        for r in logs:
            f.write(json.dumps(r) + "\n")
    print("logs →", str(out_dir))

if __name__ == "__main__":
    main()
