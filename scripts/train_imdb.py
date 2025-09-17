# scripts/train_imdb.py
import os, sys, time, json, argparse, numpy as np
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Make sure local src is importable when running from repo root or notebooks/
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from spectral_attention.models_cls import SpectralClassifier, VanillaClassifier  # noqa: E402
from spectral_attention.utils import resolve_device, parse_amp_dtype  # noqa: E402


def set_seed(s=1234):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def collate(tokenizer, max_len):
    def _fn(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc = tokenizer(texts, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_len)
        return enc["input_ids"], enc["attention_mask"], labels
    return _fn


def make_model(kind, vocab_size, d_model, heads, depth, max_len, dropout, use_dct, num_classes):
    if kind == "spectral":
        return SpectralClassifier(vocab_size, d_model, heads, depth, max_len, dropout, use_dct, num_classes)
    elif kind == "vanilla":
        return VanillaClassifier(vocab_size, d_model, heads, depth, max_len, dropout, num_classes)
    raise ValueError(kind)


def save_jsonl(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None):
    model.eval()
    lossf = nn.CrossEntropyLoss()
    tot_loss, tot_cnt, tot_acc = 0.0, 0, 0.0
    for input_ids, attn_mask, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logits = model(input_ids, attn_mask)
            loss = lossf(logits, labels)
        pred = logits.argmax(-1)
        tot_acc += (pred == labels).float().sum().item()
        tot_loss += float(loss) * labels.size(0)
        tot_cnt  += labels.size(0)
    return tot_loss / max(1, tot_cnt), tot_acc / max(1, tot_cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["spectral","vanilla"], required=True)
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--dmodel", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_dct", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--mixed_precision", choices=["bf16","fp16","none"], default="bf16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--outdir", default="experiments/runs/imdb")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[train_imdb] Using device: {device}")

    # Data + tokenizer
    ds = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.cls_token
    vocab_size = tokenizer.vocab_size

    train_loader = DataLoader(ds["train"], batch_size=args.batch, shuffle=True,
                              num_workers=2, collate_fn=collate(tokenizer, args.seq))
    test_loader  = DataLoader(ds["test"],  batch_size=args.batch, shuffle=False,
                              num_workers=2, collate_fn=collate(tokenizer, args.seq))

    # Model
    model = make_model(args.kind, vocab_size, args.dmodel, args.heads, args.depth,
                       args.seq, args.dropout, args.use_dct, num_classes=2).to(device)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, backend="inductor")
        except Exception as e:
            print("[warn] torch.compile failed; continuing in eager:", e)

    amp_dtype = parse_amp_dtype(args.mixed_precision)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / max(1, args.warmup))
    )

    outdir = Path(args.outdir) / time.strftime(f"{args.kind}_T{args.seq}_%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    metr = outdir / "metrics.jsonl"

    start_step, best_acc = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"]) ; opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0) ; best_acc = ck.get("best_acc", 0.0)

    # Training loop
    model.train()
    data_iter = iter(train_loader)
    for step in range(start_step + 1, args.steps + 1):
        try:
            input_ids, attn_mask, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            input_ids, attn_mask, labels = next(data_iter)

        input_ids = input_ids.to(device) ; labels = labels.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logits = model(input_ids, attn_mask)
            loss = lossf(logits, labels)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step() ; sched.step()

        # Log train
        if step % 50 == 0:
            save_jsonl(metr, {"event":"train","task":"imdb","step":step,"loss":float(loss)})

        # Eval
        if step % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, test_loader, device, amp_dtype)
            save_jsonl(metr, {"event":"val","task":"imdb","step":step,"val_loss":val_loss,"val_acc":val_acc})
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "best_acc": best_acc},
                           outdir / "ckpt_best.pt")

        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best_acc": best_acc},
                       outdir / f"ckpt_step{step}.pt")

    print("done ->", outdir)


if __name__ == "__main__":
    main()
