# scripts/eval_lm.py
import argparse, math, json
from pathlib import Path
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

from spectral_attention.models import SpectralLM, VanillaLM
from spectral_attention.utils import resolve_device

@torch.inference_mode()
def evaluate(model, ids, vocab_size, batch, seq, device, max_batches=100):
    lossf = nn.CrossEntropyLoss()
    model.eval().to(device)
    total, seen = 0.0, 0
    for _ in range(max_batches):
        N = ids.numel()
        ix = torch.randint(0, max(1, N - seq - 1), (batch,))
        x = torch.stack([ids[i:i+seq] for i in ix]).to(device)
        y = torch.stack([ids[i+1:i+seq+1] for i in ix]).to(device)
        logits = model(x)[:, :-1]
        loss = lossf(logits.reshape(-1, vocab_size), y[:, :-1].reshape(-1))
        total += float(loss); seen += 1
    avg = total / max(1, seen)
    ppl = math.exp(avg) if avg < 20 else float("inf")
    return avg, ppl

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["spectral","vanilla"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="wikitext-2-raw-v1")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"[eval_lm] Using device: {device}")

    ds = load_dataset("wikitext", args.dataset)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tok.model_max_length = args.seq
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

    def encode(split):
        text = "\n\n".join(ds[split]["text"])
        ids = tok(text, return_tensors=None, padding=False, truncation=False)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)
    val_ids = encode("validation"); test_ids = encode("test")
    vocab_size = tok.vocab_size

    # Recreate model skeleton and load
    from spectral_attention.models import make_model as _mm  # optional
    if args.kind == "spectral":
        dummy = SpectralLM(vocab_size)
    else:
        dummy = VanillaLM(vocab_size)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    dummy.load_state_dict(ckpt["model"])

    val_loss, val_ppl = evaluate(dummy, val_ids, vocab_size, args.batch, args.seq, device)
    test_loss, test_ppl = evaluate(dummy, test_ids, vocab_size, args.batch, args.seq, device)
    rec = {"val_loss": val_loss, "val_ppl": val_ppl, "test_loss": test_loss, "test_ppl": test_ppl}
    print(json.dumps(rec, indent=2))
