import math, time, json, os
from pathlib import Path
import torch, torch.nn as nn
from .blocks import SpectralEncoder
from .vanilla_blocks import VanillaEncoder

def make_model(kind="spectral", depth=4, d_model=512, n_heads=8, dropout=0.1, use_dct=False, device=None):
    if kind == "spectral":
        model = SpectralEncoder(depth=depth, d_model=d_model, n_heads=n_heads, use_dct=use_dct, dropout=dropout)
    elif kind == "vanilla":
        model = VanillaEncoder(depth=depth, d_model=d_model, n_heads=n_heads, dropout=dropout)
    else:
        raise ValueError(f"unknown kind={kind}")
    if device is not None:
        model = model.to(device)
    return model


@torch.inference_mode()
def measure_throughput(model, x, iters=50, warmup=10):
    device = x.device
    model.eval()
    # warmup
    for _ in range(warmup): _ = model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): _ = model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = x.shape[0] * x.shape[1] * iters
    tps = toks / dt
    peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    return tps, (dt/iters)*1000.0, peak

def lm_head(d_model, vocab):
    return nn.Linear(d_model, vocab, bias=False)

def train_tiny_lm(model, data_ids, vocab_size, steps=200, lr=3e-4, bsz=16, T=512, log_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    head = lm_head(model.layers[0].ff[0].in_features, vocab_size).to(device)
    opt = torch.optim.AdamW(list(model.parameters())+list(head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    N = data_ids.shape[0]
    def sample_batch():
        idx = torch.randint(0, N - T - 1, (bsz,))
        x = torch.stack([data_ids[i:i+T] for i in idx]).to(device)
        y = torch.stack([data_ids[i+1:i+T+1] for i in idx]).to(device)
        return x, y

    def embed(tok):  # simple one-hot embed for portability
        E = torch.nn.functional.one_hot(tok, num_classes=vocab_size).float()  # [B,T,V]
        W = torch.empty(vocab_size, model.layers[0].ff[0].in_features, device=device)
        torch.nn.init.normal_(W, std=0.02)
        # cache W? for simplicity re-create, it’s fine for small vocab
        return E @ W  # [B,T,D]

    logs = []
    for step in range(1, steps+1):
        x_tok, y_tok = sample_batch()           # [B,T]
        x = embed(x_tok)                        # [B,T,D]
        h = model(x)                            # [B,T,D]
        logits = head(h)                        # [B,T,V]
        loss = loss_fn(logits.view(-1, vocab_size), y_tok.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            with torch.no_grad():
                ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")
            logs.append({"step": step, "loss": loss.item(), "ppl": ppl})
            if log_path:
                Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(logs[-1]) + "\n")
    return logs
