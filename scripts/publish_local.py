# scripts/publish_local.py
import json, shutil, argparse
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="path like experiments/runs/lm/spectral_T4096_...")
    p.add_argument("--dest", default="artifacts/publish")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    dest = Path(args.dest) / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)

    # copy best ckpt + metrics + plots
    best = list(run_dir.glob("ckpt_best.pt"))
    if best:
        shutil.copy2(best[0], dest / "model_best.pt")
    for item in ["metrics.jsonl"]:
        src = run_dir / item
        if src.exists():
            shutil.copy2(src, dest / item)
    for png in run_dir.glob("freq_step*.png"):
        shutil.copy2(png, dest / png.name)

    # write a tiny model card
    card = {
        "model": run_dir.name,
        "source_run": str(run_dir),
        "notes": "Spectral vs Vanilla LM baseline run. Contains best checkpoint and metrics."
    }
    with open(dest / "modelcard.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print("published to", dest)
