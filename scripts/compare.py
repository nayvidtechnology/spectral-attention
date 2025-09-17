# scripts/compare.py
import argparse, time, math, json, os
from pathlib import Path
import torch, torch.nn as nn
from datasets import load_dataset

from spectral_attention.train_eval import make_model, measure_throughput

def set_seed(s=1234):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def mk_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def get_device(pref="auto"):
    """
    Resolve the execution device.
    - auto: use CUDA if available, else CPU
    - cpu:  force CPU
    - gpu:  force CUDA; error if not available
    """
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("--device gpu requested but CUDA is not available on this machine")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lm_data(name="wikitext-2-raw-v1", split="train", vocab="byte"):
    ds = load_dataset("wikitext", name)[split]["text"]
    text = "\n".join(ds)
    if vocab == "byte":
        import numpy as np
        arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        ids = torch.tensor(arr, dtype=torch.long)
        V = 256
    else:
        # tiny character vocab
        syms = sorted(set(text))
        stoi = {ch:i for i,ch in enumerate(syms)}
        ids  = torch.tensor([stoi.get(ch, 0) for ch in text], dtype=torch.long)
        V = len(syms)
    return ids, V

def batchify_lm(ids, V, B, T, D, device):
    N = ids.numel()
    idx = torch.randint(0, max(1, N-T-1), (B,))
    x_tok = torch.stack([ids[i:i+T]     for i in idx]).to(device)
    y_tok = torch.stack([ids[i+1:i+T+1] for i in idx]).to(device)
    # lightweight “embedding”: one-hot @ linear
    E = torch.nn.functional.one_hot(x_tok, num_classes=V).float() @ torch.randn(V, D, device=device)*0.02
    return E, y_tok, V

def imdb_data(split="train"):
    ds = load_dataset("imdb")[split]
    # byte-level encode for long docs
    X = [bytes(x, "utf-8") for x in ds["text"]]
    y = torch.tensor(ds["label"], dtype=torch.long)
    return X, y

def pad_or_trim(byte_seq, T):
    import numpy as np
    arr = np.frombuffer(byte_seq, dtype=np.uint8)
    if arr.size >= T: arr = arr[:T]
    else:
        pad = np.zeros(T - arr.size, dtype=np.uint8)
        arr = np.concatenate([arr, pad], 0)
    return torch.tensor(arr, dtype=torch.long)

def batchify_imdb(X, y, B, T, D, device):
    import random
    idx = [random.randrange(0, len(X)) for _ in range(B)]
    x_tok = torch.stack([pad_or_trim(X[i], T) for i in idx]).to(device)  # [B,T] 0..255
    Y = y[idx].to(device)
    V = 256
    E = torch.nn.functional.one_hot(x_tok, num_classes=V).float() @ torch.randn(V, D, device=device)*0.02
    return E, Y, V

def run_lm(kind, args, device, logdir, val_ids=None, V_vocab=None):
    model = make_model(kind, depth=args.depth, d_model=args.dmodel, n_heads=args.heads,
                       dropout=args.dropout, use_dct=args.use_dct).to(device)
    head  = nn.Linear(args.dmodel, args.vocab, bias=False).to(device)
    opt   = torch.optim.AdamW(list(model.parameters())+list(head.parameters()), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    logs = []

    # throughput (eval) at the start
    tps, ms, peak = measure_throughput(model, torch.randn(args.batch, args.seq, args.dmodel, device=device))
    logs.append({"event":"throughput", "kind":kind, "seq":args.seq, "tokens_per_s":tps, "ms_per_it":ms, "peakMB":peak})

    for step in range(1, args.steps+1):
        x, y_tok, V = args.sample_batch()
        logits = head(model(x))  # [B,T,V]
        loss = lossf(logits.view(-1, V), y_tok.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        if step % args.log_every == 0:
            ppl = math.exp(float(loss)) if loss < 20 else float("inf")
            rec = {"event":"train", "kind":kind, "step":step, "loss":float(loss), "ppl":ppl}
            logs.append(rec)

    # optional validation on byte-LM using separate split
    if val_ids is not None and V_vocab is not None:
        model.eval()
        total_loss, seen = 0.0, 0
        with torch.no_grad():
            for _ in range(50):
                vx, vy_tok, _ = batchify_lm(val_ids, V_vocab, args.batch, args.seq, args.dmodel, device)
                vlogits = head(model(vx))
                vloss = lossf(vlogits.view(-1, V_vocab), vy_tok.view(-1))
                total_loss += float(vloss); seen += 1
        model.train()
        vavg = total_loss / max(1, seen)
        vppl = math.exp(vavg) if vavg < 20 else float("inf")
        logs.append({"event":"val", "kind":kind, "seq":args.seq,
                     "val_loss": vavg, "val_ppl": vppl,
                     "loss": vavg, "ppl": vppl})

    mk_dir(logdir);
    with open(Path(logdir)/"metrics.jsonl", "a", encoding="utf-8") as f:
        for r in logs: f.write(json.dumps(r)+"\n")
    return logs

def run_clf(kind, args, device, logdir, X_val=None, Y_val=None):
    model = make_model(kind, depth=args.depth, d_model=args.dmodel, n_heads=args.heads,
                       dropout=args.dropout, use_dct=args.use_dct).to(device)
    head  = nn.Linear(args.dmodel, 2, bias=False).to(device)  # IMDB: 2 classes
    opt   = torch.optim.AdamW(list(model.parameters())+list(head.parameters()), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    logs = []

    tps, ms, peak = measure_throughput(model, torch.randn(args.batch, args.seq, args.dmodel, device=device))
    logs.append({"event":"throughput", "kind":kind, "seq":args.seq, "tokens_per_s":tps, "ms_per_it":ms, "peakMB":peak})

    for step in range(1, args.steps+1):
        x, Y, _ = args.sample_batch()
        logits = head(model(x)).mean(dim=1)  # [B,2] simple pooling via mean over T
        loss = lossf(logits, Y)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        if step % args.log_every == 0:
            pred = logits.argmax(-1)
            acc = (pred == Y).float().mean().item()
            rec = {"event":"train", "kind":kind, "step":step, "loss":float(loss), "acc":acc}
            logs.append(rec)

    # optional validation on IMDB test split
    if X_val is not None and Y_val is not None:
        model.eval()
        total_loss, total_acc, seen = 0.0, 0.0, 0
        with torch.no_grad():
            for _ in range(50):
                vx, vY, _ = batchify_imdb(X_val, Y_val, args.batch, args.seq, args.dmodel, device)
                vlogits = head(model(vx)).mean(dim=1)
                vloss = lossf(vlogits, vY)
                pred = vlogits.argmax(-1)
                vacc = (pred == vY).float().mean().item()
                total_loss += float(vloss); total_acc += float(vacc); seen += 1
        model.train()
        vavg_loss = total_loss / max(1, seen)
        vavg_acc  = total_acc  / max(1, seen)
        logs.append({"event":"val", "kind":kind, "seq":args.seq,
                     "val_loss": vavg_loss, "loss": vavg_loss,
                     "val_acc": vavg_acc,  "acc": vavg_acc})

    mk_dir(logdir)
    with open(Path(logdir)/"metrics.jsonl", "a", encoding="utf-8") as f:
        for r in logs: f.write(json.dumps(r)+"\n")
    return logs

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["lm", "imdb"], default="lm")
    p.add_argument("--kind", choices=["spectral", "vanilla"], required=True)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dmodel", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_dct", action="store_true")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    p.add_argument("--logdir", type=str, default="experiments/runs/compare")
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()

    set_seed(1234)
    device = get_device(args.device)
    print(f"[compare] Using device: {device}")

    if args.task == "lm":
        # train and validation splits
        ids, V = lm_data("wikitext-2-raw-v1", split="train", vocab="byte")
        val_ids, Vv = lm_data("wikitext-2-raw-v1", split="validation", vocab="byte")
        assert V == Vv, "Train/val vocab mismatch"
        args.vocab = V
        args.sample_batch = lambda: batchify_lm(ids, V, args.batch, args.seq, args.dmodel, device)
        run_fn = lambda k,a,d,ld: run_lm(k, a, d, ld, val_ids=val_ids, V_vocab=V)
    else:
        X_tr, Y_tr = imdb_data("train")
        X_te, Y_te = imdb_data("test")
        args.vocab = 256
        args.sample_batch = lambda: batchify_imdb(X_tr, Y_tr, args.batch, args.seq, args.dmodel, device)
        run_fn = lambda k,a,d,ld: run_clf(k, a, d, ld, X_val=X_te, Y_val=Y_te)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.logdir, f"{args.task}_{args.kind}_T{args.seq}_{stamp}")
    run_fn(args.kind, args, device, out)
    print("logs →", out)
