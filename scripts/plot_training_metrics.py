import argparse, json, math, pathlib
from typing import List, Dict, Any
import matplotlib.pyplot as plt

METRIC_KEYS = [
    ("loss", "Loss"),
    ("ce", "Cross Entropy"),
    ("ppl", "Perplexity"),
    ("lr", "Learning Rate"),
    ("alpha", "KD Alpha"),
    ("kd_loss", "KD Loss"),
    ("hidden_mse", "Hidden MSE"),
    ("spec_smooth", "Spectral Smoothness"),
    ("cum_tokens", "Cumulative Tokens"),
]


def load_metrics(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def filter_events(rows, event="train"):
    return [r for r in rows if r.get("event") == event]


def plot_metrics(rows: List[Dict[str, Any]], out_dir: pathlib.Path, title_prefix: str = ""):
    if not rows:
        print("No training rows to plot.")
        return
    steps = [r["step"] for r in rows if "step" in r]
    for key, label in METRIC_KEYS:
        if key not in rows[0]:
            # skip if metric absent in first; may appear later but avoid sparse axes noise
            present = any(key in r for r in rows)
            if not present:
                continue
        xs, ys = [], []
        for r in rows:
            if key in r and "step" in r:
                xs.append(r["step"])
                val = r[key]
                if key == "ppl" and (not math.isfinite(val) or val > 1e5):
                    continue
                ys.append(val)
        if not xs:
            continue
        plt.figure(figsize=(6,4))
        plt.plot(xs, ys, marker='o', ms=2, linewidth=1)
        plt.xlabel("Step")
        plt.ylabel(label)
        plt.title(f"{title_prefix}{label}")
        plt.grid(alpha=0.3)
        out_path = out_dir / f"{key}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[plot] saved {out_path}
")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=str, help="Path to metrics.jsonl produced by training script")
    ap.add_argument("--out", type=str, default=None, help="Output directory for plots (default: metrics parent /plots)")
    ap.add_argument("--prefix", type=str, default="", help="Title prefix")
    args = ap.parse_args()

    metrics_path = pathlib.Path(args.metrics)
    rows = load_metrics(metrics_path)
    train_rows = filter_events(rows, "train")

    if args.out:
        out_dir = pathlib.Path(args.out)
    else:
        out_dir = metrics_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics(train_rows, out_dir, title_prefix=args.prefix)

    # Simple aggregate summary
    if train_rows:
        final = train_rows[-1]
        summary_keys = [k for k,_ in METRIC_KEYS if k in final]
        print("\nFinal step summary:")
        for k in summary_keys:
            print(f"  {k}: {final[k]}")

if __name__ == "__main__":
    main()
