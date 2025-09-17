# scripts/eval_imdb.py
import argparse, json, sys
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

# Ensure local src on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from spectral_attention.models_cls import SpectralClassifier, VanillaClassifier  # noqa: E402
from spectral_attention.utils import resolve_device  # noqa: E402


def collate(tokenizer, max_len):
    def _fn(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc = tokenizer(texts, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_len)
        return enc["input_ids"], enc["attention_mask"], labels
    return _fn


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval().to(device)
    lossf = nn.CrossEntropyLoss()
    tot_loss, tot_cnt, tot_acc = 0.0, 0, 0.0
    for input_ids, attn_mask, labels in loader:
        input_ids = input_ids.to(device); labels = labels.to(device)
        logits = model(input_ids, attn_mask)
        loss = lossf(logits, labels)
        pred = logits.argmax(-1)
        tot_acc += (pred == labels).float().sum().item()
        tot_loss += float(loss) * labels.size(0)
        tot_cnt  += labels.size(0)
    return tot_loss / max(1, tot_cnt), tot_acc / max(1, tot_cnt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["spectral","vanilla"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"[eval_imdb] Using device: {device}")

    ds = load_dataset("imdb")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.cls_token
    vocab_size = tok.vocab_size

    loader = DataLoader(ds["test"], batch_size=args.batch, shuffle=False,
                        num_workers=2, collate_fn=collate(tok, args.seq))

    if args.kind == "spectral":
        model = SpectralClassifier(vocab_size)
    else:
        model = VanillaClassifier(vocab_size)

    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])

    loss, acc = evaluate(model, loader, device)
    print(json.dumps({"loss": loss, "acc": acc}, indent=2))
