import argparse, json, math, os, time
from pathlib import Path

import torch
from datasets import load_dataset
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from spectral_attention import convert_gpt2lm_to_spectral
from spectral_attention.utils import resolve_device


def set_seed(s=1234):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.inference_mode()
def measure_throughput(model, x_ids, attn_mask, iters=30, warmup=10):
    model.eval()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(input_ids=x_ids, attention_mask=attn_mask)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(input_ids=x_ids, attention_mask=attn_mask)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = x_ids.numel()
    return (toks * iters) / dt, (dt / iters) * 1000.0, (torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.0


def tokenize_wikitext(tokenizer, block_size=1024):
    raw_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    raw_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tok_fn(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=False, truncation=False, padding=False)

    train_tok = raw_train.map(tok_fn, batched=True, remove_columns=["text"])
    val_tok = raw_val.map(tok_fn, batched=True, remove_columns=["text"])

    # group texts into block_size chunks
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    train_ds = train_tok.map(group_texts, batched=True)
    val_ds = val_tok.map(group_texts, batched=True)
    return train_ds, val_ds


class JsonlLoggerCallback:
    def __init__(self, jsonl_path: Path, kind: str, seq: int):
        self.path = jsonl_path
        self.kind = kind
        self.seq = seq

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        rec = {"event": "train", "kind": self.kind, "step": int(state.global_step), "seq": self.seq}
        if "loss" in logs:
            rec["loss"] = float(logs["loss"])  # trainer reports running loss
            if rec["loss"] < 20:
                rec["ppl"] = math.exp(rec["loss"])  # approximate
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics: return
        rec = {"event": "val", "kind": self.kind, "seq": self.seq}
        if "eval_loss" in metrics:
            v = float(metrics["eval_loss"])
            rec["val_loss"] = v
            rec["loss"] = v
            rec["val_ppl"] = math.exp(v) if v < 20 else float("inf")
            rec["ppl"] = rec["val_ppl"]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--spectral", action="store_true", help="Swap GPT-2 attention with SpectralAttention")
    ap.add_argument("--use_dct", action="store_true")
    ap.add_argument("--token_gate", action="store_true")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--logdir", type=str, default="experiments/runs/hf_trainer")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    args = ap.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = tokenize_wikitext(tokenizer, block_size=args.seq)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    cfg = AutoConfig.from_pretrained(args.model_name)
    kind = f"hf_{args.model_name}{'_spectral' if args.spectral else ''}"
    if args.spectral:
        model = convert_gpt2lm_to_spectral(cfg, opts=None, from_pretrained=args.model_name)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_name, config=cfg)

    # Throughput probe (synthetic)
    device = resolve_device(args.device)
    print(f"[train_lm_hf_trainer] Using device: {device}")
    model = model.to(device)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    synth = torch.full((args.per_device_train_batch_size, args.seq), tokenizer.eos_token_id, device=device, dtype=torch.long)
    attn = (synth != tokenizer.pad_token_id)
    tok_s, ms, peakMB = measure_throughput(model, synth, attn)

    # Prepare output dir and metrics log
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.logdir) / f"lm_{kind}_T{args.seq}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "metrics.jsonl"
    with open(jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "throughput", "task": "lm", "kind": kind, "seq": args.seq,
            "tokens_per_s": tok_s, "ms_per_it": ms, "peakMB": peakMB
        }) + "\n")

    args_tr = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=10_000,
        logging_steps=max(1, args.eval_steps // 2),
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],  # disable WandB, etc.
        push_to_hub=False,
        remove_unused_columns=False,
    )

    cb = JsonlLoggerCallback(jsonl, kind, args.seq)

    trainer = Trainer(
        model=model,
        args=args_tr,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        callbacks=[cb],
    )

    trainer.train()
    print("logs →", str(out_dir))


if __name__ == "__main__":
    main()
