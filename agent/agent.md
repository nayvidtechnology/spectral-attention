# Agent Tasking Guide

Goal: Implement, test, and benchmark a **Spectral (FFT/DCT) Attention** global mixer that is stable, efficient, and interpretable. Must run on **CPU or GPU** automatically. Python **3.10.18**.

## Environment
- Prefer conda: `conda env create -f env/conda-env.yaml && conda activate spectral-attn`
- CPU/GPU: detect with `torch.cuda.is_available()`

## Tasks
1. **Code**
   - Implement `src/spectral_attn/spectral_attention.py` using rFFT/irFFT or DCT-II.
   - Ensure residual connections, dropout, and lazy bin init (depends on seq length).
   - Add `SpectralEncoderBlock` in `src/spectral_attn/blocks.py`.

2. **Tests**
   - Run `pytest`. Fix any shape/grad/AMP issues.
   - Guarantee output shape equals input shape.

3. **Bench & Smoke**
   - Run `python scripts/bench_spectral_attention.py` for tokens/s.
   - Run `python scripts/smoke_train.py` to verify learning & gradient stability.

4. **Docs**
   - Update `docs/design.md` with math, stability tricks (Hermitian tying, rFFT bins).
   - Update `docs/api.md` with public classes/functions.
   - Promote green experiment results from `experiments/promote` → `docs/experiments.md`.

5. **Experiments**
   - Save each run under `experiments/runs/<ISO8601_name>/`.
   - Include `hparams.json`, `metrics.jsonl`, and plots in `artifacts/`.

## Coding Conventions
- Use `torch.fft.rfft/irfft` for complex path; `torch.fft.dct/idct` for real path.
- Keep spectral params float32 even under AMP; cast IO back to input dtype.
- Avoid quadratic ops; keep token-gating rank-1 or off by default.

## Success Criteria
- No NaNs under AMP.
- ≥ Vanilla-attention throughput on seq-4k.
- Clean unit tests; reproducible perf numbers logged.
