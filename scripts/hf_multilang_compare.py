import argparse, json, math, time, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config

# Ensure local src is importable when running from repo root or notebooks/
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from spectral_attention.utils import resolve_device, parse_amp_dtype  # noqa: E402
from spectral_attention import convert_gpt2lm_to_spectral  # noqa: E402


def set_seed(s=1234):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def prompt_nll(model, tok, text_ids: torch.Tensor) -> float:
    """Compute average negative log-likelihood (cross-entropy) on the prompt itself."""
    # labels: shifted by one; ignore first position
    labels = text_ids.clone()
    labels[:, :-1] = text_ids[:, 1:]
    labels[:, -1] = -100
    attn = (text_ids != tok.pad_token_id)
    out = model(input_ids=text_ids, attention_mask=attn)
    logits = out.logits
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    loss = lossf(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    return float(loss)


def decode_tokens(tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model", default="gpt2")
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="bf16")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--logdir", type=str, default="experiments/runs/hf_multilang_compare")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--spectral_ckpt", type=str, default=None, help="Optional path to spectral state_dict checkpoint (.pt) to load")
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[hf_multilang_compare] Using device: {device}")
    dtype = parse_amp_dtype(args.dtype) or torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Multilingual prompts (English, Hindi, Gujarati, Kannada)
    prompts = [
        ("en", "The quick brown fox jumps over the lazy dog."),
        ("en", "In a shocking finding, scientists discovered that"),
        ("hi", "भारत के संविधान की प्रस्तावना कहती है कि भारत एक"),
        ("gu", "ભારતનું બંધારણ કહે છે કે ભારત એક"),
        ("kn", "ಭಾರತದ ಸಂವಿಧಾನದ ಮುನ್ನುಡಿ ಹೇಳುತ್ತದೆ ಭಾರತ ಒಂದು"),
    ]

    # Prepare models: vanilla GPT-2 and Spectral GPT-2 (attention swapped)
    # Vanilla
    vanilla = AutoModelForCausalLM.from_pretrained(args.hf_model)
    try:
        vanilla = vanilla.to(device).to(dtype)
    except Exception:
        vanilla = vanilla.to(device)  # fallback to fp32 if dtype not supported
    vanilla.eval()

    # Spectral (load config and weights from the same base model)
    cfg = GPT2Config.from_pretrained(args.hf_model)
    spectral = convert_gpt2lm_to_spectral(cfg, from_pretrained=args.hf_model)
    # Optionally load a fine-tuned spectral checkpoint
    if args.spectral_ckpt:
        ckpt_path = Path(args.spectral_ckpt)
        if ckpt_path.is_file():
            sd = torch.load(str(ckpt_path), map_location=device)
            missing, unexpected = spectral.load_state_dict(sd, strict=False)
            print(f"[spectral_ckpt] loaded {ckpt_path.name} | missing: {len(missing)} | unexpected: {len(unexpected)}")
        else:
            print(f"[spectral_ckpt] WARNING: not found at {args.spectral_ckpt}")
    try:
        spectral = spectral.to(device).to(dtype)
    except Exception:
        spectral = spectral.to(device)
    spectral.eval()

    # For logging results
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.logdir) / f"multilang_{args.hf_model}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / "outputs.jsonl"

    def encode(txt: str) -> torch.Tensor:
        enc = tokenizer([txt], return_tensors="pt", padding=True)
        return enc["input_ids"].to(device)

    def generate(m, ids: torch.Tensor) -> torch.Tensor:
        attn = (ids != tokenizer.pad_token_id)
        gen = m.generate(
            input_ids=ids,
            attention_mask=attn,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return gen

    for lang, txt in prompts:
        ids = encode(txt)
        # NLL on the prompt
        v_loss = prompt_nll(vanilla, tokenizer, ids)
        s_loss = prompt_nll(spectral, tokenizer, ids)
        v_ppl = math.exp(v_loss) if v_loss < 20 else float("inf")
        s_ppl = math.exp(s_loss) if s_loss < 20 else float("inf")

        # Short continuations
        v_out = generate(vanilla, ids)
        s_out = generate(spectral, ids)
        v_text = decode_tokens(tokenizer, v_out)
        s_text = decode_tokens(tokenizer, s_out)

        rec = {
            "lang": lang,
            "prompt": txt,
            "vanilla": {"loss": v_loss, "ppl": v_ppl, "text": v_text},
            "spectral": {"loss": s_loss, "ppl": s_ppl, "text": s_text},
        }
        print("\n==>", lang)
        print("prompt:", txt)
        print("-- vanilla -- loss=", v_loss, "ppl=", v_ppl)
        print(v_text)
        print("-- spectral -- loss=", s_loss, "ppl=", s_ppl)
        print(s_text)
        with open(outp, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("logs →", str(out_dir))


if __name__ == "__main__":
    main()
