# scripts/compare_suite.py
import subprocess, time, json, os, sys
from pathlib import Path

def run(cmd):
    print(">>", " ".join(cmd))
    return subprocess.run(cmd, check=True)

if __name__ == "__main__":
    outroot = Path("experiments/runs/compare_suite") / time.strftime("%Y%m%d-%H%M%S")
    outroot.mkdir(parents=True, exist_ok=True)
    seqs = [1024, 2048, 4096]
    steps = 800

    for T in seqs:
        for kind in ["spectral", "vanilla"]:
            cmd = [
                sys.executable, "scripts/train_lm.py",
                "--kind", kind, "--seq", str(T),
                "--steps", str(steps), "--device", "gpu",
                "--mixed_precision", "bf16", "--outdir", str(outroot),
            ]
            run(cmd)
