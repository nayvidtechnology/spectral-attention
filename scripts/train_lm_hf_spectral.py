import argparse, time, math, json, torch
from pathlib import Path
from transformers import AutoTokenizer, GPT2Config, AutoModelForCausalLM

from spectral_attention import convert_gpt2lm_to_spectral
from spectral_attention.utils import resolve_device, parse_amp_dtype


def set_seed(s=1234):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.inference_mode()
def measure_throughput(model, x, iters=50, warmup=10):
    model.eval()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    for _ in range(warmup): _ = model(input_ids=x, attention_mask=(x != pad_id))
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(input_ids=x, attention_mask=(x != pad_id))
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = x.numel()
    return (toks * iters) / dt, (dt / iters) * 1000.0, (torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.0


def load_wikitext_ids(tokenizer, dataset="wikitext-2-raw-v1", split="train", allow_synth=False):
    """Load the wikitext dataset into a flat tensor of token ids.

    Fails fast if dataset cannot be loaded unless allow_synth=True, in which case
    a synthetic random token stream (2M tokens) is returned. Adds an EOS token
    between documents to reduce cross-document bleed.
    """
    try:
        from datasets import load_dataset  # lazy import
        ds = load_dataset("wikitext", dataset)[split]["text"]
        all_ids = []
        eos_id = tokenizer.eos_token_id
        for doc in ds:
            if not doc:
                continue
            enc = tokenizer(doc, return_tensors=None, truncation=False, padding=False)["input_ids"]
            all_ids.extend(enc + [eos_id])
        if not all_ids:
            raise RuntimeError("Loaded dataset was empty after tokenization")
        return torch.tensor(all_ids, dtype=torch.long)
    except Exception as e:
        if not allow_synth:
            raise RuntimeError(f"Dataset load failed and synthetic fallback disabled: {e}") from e
        print(f"[data] WARNING: dataset load failed; using synthetic tokens. reason={e}")
        vocab_size = getattr(tokenizer, "vocab_size", 50257)
        return torch.randint(0, vocab_size, (2_000_000,), dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model", default="gpt2", help="base model to load embeddings/MLP/LayerNorm from")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--seq_schedule", type=str, default=None, help="Comma-separated sequence lengths to progress through (e.g. 512,1024,1536,2048)")
    ap.add_argument("--seq_schedule_steps", type=int, default=0, help="Number of steps to train at each scheduled seq before advancing (0=auto distribute)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=50, help="Linear warmup steps for LR (after which cosine decay begins)")
    ap.add_argument("--min_lr", type=float, default=1e-5, help="Minimum LR for cosine scheduler")
    ap.add_argument("--wd", type=float, default=0.1, help="Weight decay for non-spectral params")
    ap.add_argument("--max_grad_norm", type=float, default=1.0, help="Clip gradient norm (0 disables)")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="bf16")
    ap.add_argument("--device", choices=["auto","cpu","gpu"], default="auto")
    ap.add_argument("--use_dct", action="store_true")
    ap.add_argument("--token_gate", action="store_true")
    ap.add_argument("--logdir", type=str, default="experiments/runs/compare")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--allow_synthetic", action="store_true", help="Allow synthetic random token fallback if dataset load fails (otherwise fail fast)")
    ap.add_argument("--require_env", type=str, default=None, help="Fail if CONDA_DEFAULT_ENV or basename of VIRTUAL_ENV does not match this value")
    # Distillation and freezing options
    ap.add_argument("--kd", action="store_true", help="Enable knowledge distillation from a vanilla teacher")
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="Weight of KD loss vs CE loss")
    ap.add_argument("--kd_tau", type=float, default=2.0, help="Distillation temperature")
    ap.add_argument("--teacher_model", type=str, default=None, help="HF model name for teacher (defaults to --hf_model)")
    ap.add_argument("--freeze_non_attn", action="store_true", help="Freeze non-attention weights for a warmup phase")
    # KD scheduling
    ap.add_argument("--kd_warmup_steps", type=int, default=0, help="Steps with alpha=0 before ramp")
    ap.add_argument("--kd_ramp_steps", type=int, default=100, help="Steps to linearly increase alpha to target")
    ap.add_argument("--kd_decay_steps", type=int, default=0, help="Last steps to linearly decay alpha back to 0; 0 disables")
    # Spectral-specific optimizer tuning
    ap.add_argument("--spectral_lr_scale", type=float, default=2.0, help="Multiply LR for spectral params by this factor")
    ap.add_argument("--spectral_weight_decay", type=float, default=0.01, help="Weight decay for spectral params")
    # Checkpointing
    ap.add_argument("--save_ckpt", type=str, default=None, help="Path to save model state_dict at end of run (.pt)")
    ap.add_argument("--resume_ckpt", type=str, default=None, help="Path to load model state_dict before training (.pt)")
    # Hidden-state / spectral regularization (P2)
    ap.add_argument("--hidden_mse", action="store_true", help="Add hidden-state MSE distillation loss (teacher last hidden vs student)")
    ap.add_argument("--hidden_mse_weight", type=float, default=0.1, help="Weight for hidden-state MSE auxiliary loss")
    ap.add_argument("--spec_smooth", action="store_true", help="Enable spectral smoothness regularizer (finite diff on log_gain/phase)")
    ap.add_argument("--spec_smooth_weight", type=float, default=1e-4, help="Weight for spectral smoothness penalty")
    ap.add_argument("--grad_checkpoint", action="store_true", help="Enable gradient (activation) checkpointing to reduce memory usage")
    ap.add_argument("--no_spectral_smart_init", action="store_true", help="Disable smart spectral initialization (use zeros)")
    args = ap.parse_args()

    if args.require_env:
        import os
        env_name = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV", "")
        base = os.path.basename(env_name) if env_name else env_name
        if not env_name or (args.require_env not in (env_name, base)):
            raise SystemExit(f"[env] Required environment '{args.require_env}' not active (found '{env_name}')")

    set_seed(1234)
    device = resolve_device(args.device)
    print(f"[train_lm_hf_spectral] Using device: {device}")
    dtype = parse_amp_dtype(args.dtype) or torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global pad_id
    pad_id = tokenizer.pad_token_id

    # Build spectral GPT-2 LMHead model (load pretrained for non-attn weights)
    if args.no_spectral_smart_init:
        import os
        os.environ["SPECTRAL_NO_SMART_INIT"] = "1"
    cfg = GPT2Config.from_pretrained(args.hf_model)
    model = convert_gpt2lm_to_spectral(cfg, from_pretrained=args.hf_model)
    model = model.to(device).to(dtype)
    model.train()

    if args.grad_checkpoint:
        # HuggingFace models expose gradient_checkpointing_enable; our converted model should retain same interface
        if hasattr(model, 'gradient_checkpointing_enable'):
            try:
                model.gradient_checkpointing_enable()
                print("[memory] Gradient checkpointing enabled")
            except Exception as e:
                print(f"[memory] WARNING: could not enable gradient checkpointing: {e}")
        else:
            print("[memory] WARNING: model lacks gradient_checkpointing_enable; skipping")

    # Build optimizer with param groups
    spectral_params = []
    base_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "spectral" in n.lower():
            spectral_params.append(p)
        else:
            base_params.append(p)
    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr, "weight_decay": args.wd})
    if spectral_params:
        param_groups.append({
            "params": spectral_params,
            "lr": args.lr * args.spectral_lr_scale,
            "weight_decay": args.spectral_weight_decay,
        })
    if not param_groups:  # fallback (e.g., all frozen)
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.wd}]
    opt = torch.optim.AdamW(param_groups)
    # store initial learning rates per param group for scheduling
    for pg in opt.param_groups:
        pg["initial_lr"] = pg["lr"]
    lossf = torch.nn.CrossEntropyLoss()
    # Optional: resume from checkpoint
    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        if ckpt_path.is_file():
            sd = torch.load(str(ckpt_path), map_location=device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[resume] loaded: {ckpt_path.name} | missing: {len(missing)} | unexpected: {len(unexpected)}")
        else:
            print(f"[resume] WARNING: checkpoint not found at {args.resume_ckpt}")

    # Optional: freeze non-attention weights for a warmup
    if args.freeze_non_attn:
        frozen = unfrozen = 0
        for n, p in model.named_parameters():
            key = n.lower()
            if ("attn" in key) or ("spectral" in key) or ("attention" in key):
                p.requires_grad = True; unfrozen += p.numel()
            else:
                p.requires_grad = False; frozen += p.numel()
        print(f"[freeze] frozen params: {frozen:,} | trainable (attn/spectral): {unfrozen:,}")

    # Optional: initialize teacher for knowledge distillation
    teacher = None
    if args.kd:
        t_name = args.teacher_model or args.hf_model
        teacher = AutoModelForCausalLM.from_pretrained(t_name).to(device)
        teacher.eval()

    # synthetic input for throughput
    x_synth = torch.full((args.batch, args.seq), fill_value=tokenizer.eos_token_id, device=device, dtype=torch.long)
    tok_s, ms, peakMB = measure_throughput(model, x_synth)

    # dataset ids
    ids = load_wikitext_ids(tokenizer, split="train", allow_synth=args.allow_synthetic).to(device)
    print(f"[data] token_corpus_size={ids.numel():,} synthetic={'yes' if args.allow_synthetic else 'no'}")

    def sample_batch():
        N = ids.numel()
        idx = torch.randint(0, max(1, N-args.seq-1), (args.batch,), device=device)
        x = torch.stack([ids[i:i+args.seq] for i in idx])
        y = torch.stack([ids[i+1:i+args.seq+1] for i in idx])
        return x, y

    logs = []
    kind = f"hf_{args.hf_model}_spectral"
    logs.append({"event":"throughput", "task":"lm", "kind": kind, "seq": args.seq,
                 "tokens_per_s": tok_s, "ms_per_it": ms, "peakMB": peakMB})

    cumulative_tokens = 0
    # Curriculum sequence schedule preparation
    if args.seq_schedule:
        schedule = [int(s) for s in args.seq_schedule.split(',') if s.strip()]
        if args.seq not in (schedule[-1], schedule[0]):
            # ensure final seq corresponds to provided --seq for logging consistency
            schedule.append(args.seq)
        # remove duplicates while preserving order
        seen = set(); ordered = []
        for s in schedule:
            if s not in seen:
                ordered.append(s); seen.add(s)
        schedule = ordered
    else:
        schedule = [args.seq]
    if args.seq_schedule_steps > 0 and len(schedule) > 1:
        steps_per_stage = [args.seq_schedule_steps] * len(schedule)
    else:
        # Distribute steps evenly across stages
        base = args.steps // len(schedule)
        steps_per_stage = [base] * len(schedule)
        steps_per_stage[-1] += args.steps - base * len(schedule)
    stage_bounds = []
    acc = 0
    for sp, seq_len in zip(steps_per_stage, schedule):
        stage_bounds.append((acc, acc + sp, seq_len))  # [start_step, end_step, seq]
        acc += sp
    current_seq = schedule[0]

    def current_seq_for_step(step: int):
        for start, end, sl in stage_bounds:
            if start < step <= end:
                return sl
        return stage_bounds[-1][2]

    # interpolation helper for spectral params when sequence grows
    def interpolate_spectral_params(model, old_T, new_T):
        if new_T <= old_T:
            return
        # For rFFT path bins = T//2+1; for DCT path bins=T
        def interp_param(p, is_dct):
            if p is None:
                return p
            old_bins = p.size(1)
            new_bins = new_T if is_dct else (new_T // 2 + 1)
            if new_bins == old_bins:
                return p
            old_idx = torch.linspace(0, 1, old_bins, device=p.device)
            new_idx = torch.linspace(0, 1, new_bins, device=p.device)
            expanded = []
            for head in p:
                expanded.append(torch.interp(new_idx, old_idx, head))
            return torch.stack(expanded, dim=0)
        for mod in model.modules():
            if hasattr(mod, 'log_gain') and hasattr(mod, '_initialized_bins') and int(getattr(mod, '_initialized_bins').item()) > 0:
                is_dct = getattr(mod, 'use_dct', False)
                new_log = interp_param(mod.log_gain.data if mod.log_gain is not None else None, is_dct)
                new_phase = interp_param(mod.phase.data if mod.phase is not None else None, is_dct)
                if new_log is not None:
                    mod.log_gain = torch.nn.Parameter(new_log)
                if new_phase is not None:
                    mod.phase = torch.nn.Parameter(new_phase)
                # update tracker
                bins_now = new_T if is_dct else (new_T // 2 + 1)
                mod._initialized_bins = torch.tensor(bins_now, device=mod.log_gain.device if mod.log_gain is not None else device)

    last_seq = current_seq
    def adjust_lr(step: int):
        # cosine schedule after warmup
        if step <= args.warmup_steps and args.warmup_steps > 0:
            warm_frac = step / max(1, args.warmup_steps)
            for pg in opt.param_groups:
                base = pg["initial_lr"]
                scale = base * warm_frac
                # preserve spectral scale ratio by basing on its own initial_lr
                pg["lr"] = scale
            return
        # post-warmup cosine
        progress = (step - args.warmup_steps) / max(1, (args.steps - args.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups:
            base = pg["initial_lr"]
            pg["lr"] = args.min_lr + (base - args.min_lr) * cos_factor

    def kd_alpha_at(step: int):
        if not args.kd:
            return 0.0
        # warmup (alpha=0)
        if step <= args.kd_warmup_steps:
            return 0.0
        # ramp
        ramp_end = args.kd_warmup_steps + args.kd_ramp_steps
        if step <= ramp_end and args.kd_ramp_steps > 0:
            frac = (step - args.kd_warmup_steps) / max(1, args.kd_ramp_steps)
            return args.kd_alpha * frac
        # decay at tail
        if args.kd_decay_steps > 0 and step > (args.steps - args.kd_decay_steps):
            frac_tail = (args.steps - step) / max(1, args.kd_decay_steps)
            return args.kd_alpha * max(0.0, frac_tail)
        return args.kd_alpha

    for step in range(1, args.steps+1):
        adjust_lr(step)
        # Possibly advance curriculum
        target_seq = current_seq_for_step(step)
        if target_seq != current_seq:
            # interpolate params then update sequence-related state
            interpolate_spectral_params(model, current_seq, target_seq)
            current_seq = target_seq
            last_seq = current_seq
        # re-sample batch with current sequence length (adjust sample_batch logic)
        if current_seq != args.seq:
            # temporary override of sequence for this stage
            seq_len = current_seq
            def sample_curriculum_batch():
                N = ids.numel()
                idx = torch.randint(0, max(1, N-seq_len-1), (args.batch,), device=device)
                x_local = torch.stack([ids[i:i+seq_len] for i in idx])
                y_local = torch.stack([ids[i+1:i+seq_len+1] for i in idx])
                return x_local, y_local
            x,y = sample_curriculum_batch()
        else:
            x,y = sample_batch()
        attn = (x != pad_id)
        out = model(input_ids=x, attention_mask=attn)
        logits = out.logits[:, :-1]
        student_hidden = getattr(out, 'last_hidden_state', None)
        ce = lossf(logits.reshape(-1, logits.size(-1)), y[:, :-1].reshape(-1))

        kd_loss = torch.tensor(0.0, device=device)
        hidden_mse = torch.tensor(0.0, device=device)
        if teacher is not None:
            with torch.no_grad():
                t_out = teacher(input_ids=x, attention_mask=attn)
                t_logits = t_out.logits[:, :-1]
            tau = float(args.kd_tau)
            log_p = torch.log_softmax(logits / tau, dim=-1)
            q = torch.softmax(t_logits / tau, dim=-1)
            kd_loss = torch.nn.functional.kl_div(log_p, q, reduction="batchmean") * (tau * tau)
            if args.hidden_mse and student_hidden is not None and hasattr(t_out, 'last_hidden_state'):
                hidden_mse = torch.mean((student_hidden - t_out.last_hidden_state)**2)

        alpha = kd_alpha_at(step) if teacher is not None else 0.0
        # spectral smoothness on log_gain / phase (finite difference L2)
        spec_smooth_loss = torch.tensor(0.0, device=device)
        if args.spec_smooth:
            for mod in model.modules():
                if hasattr(mod, 'log_gain') and isinstance(getattr(mod, 'log_gain'), torch.nn.Parameter):
                    lg = mod.log_gain
                    if lg is not None and lg.ndim == 2 and lg.size(1) > 2:
                        spec_smooth_loss = spec_smooth_loss + (lg[:,1:] - lg[:,:-1]).pow(2).mean()
                if hasattr(mod, 'phase') and isinstance(getattr(mod, 'phase'), torch.nn.Parameter):
                    ph = mod.phase
                    if ph is not None and ph.ndim == 2 and ph.size(1) > 2:
                        spec_smooth_loss = spec_smooth_loss + (ph[:,1:] - ph[:,:-1]).pow(2).mean()
        loss = alpha * kd_loss + (1.0 - alpha) * ce
        if args.hidden_mse:
            loss = loss + args.hidden_mse_weight * hidden_mse
        if args.spec_smooth:
            loss = loss + args.spec_smooth_weight * spec_smooth_loss

        opt.zero_grad(set_to_none=True); loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        cumulative_tokens += x.numel()
        if step % args.log_every == 0:
            ppl = math.exp(float(ce)) if ce < 20 else float("inf")
            current_lr = opt.param_groups[0]["lr"] if opt.param_groups else args.lr
            rec = {"event":"train", "kind": kind, "step": step, "loss": float(loss), "ppl": ppl, "ce": float(ce), "lr": current_lr}
            if teacher is not None:
                rec.update({"kd_loss": float(kd_loss), "alpha": alpha, "tau": float(args.kd_tau)})
            if args.hidden_mse:
                rec["hidden_mse"] = float(hidden_mse)
            if args.spec_smooth:
                rec["spec_smooth"] = float(spec_smooth_loss)
            rec["cum_tokens"] = cumulative_tokens
            logs.append(rec)

    # quick validation
    with torch.no_grad():
        total, seen = 0.0, 0
        for _ in range(50):
            x,y = sample_batch()
            attn = (x != pad_id)
            out = model(input_ids=x, attention_mask=attn)
            logits = out.logits[:, :-1]
            loss = lossf(logits.reshape(-1, logits.size(-1)), y[:, :-1].reshape(-1))
            total += float(loss); seen += 1
        vavg = total / max(1, seen)
        vppl = math.exp(vavg) if vavg < 20 else float("inf")
        logs.append({"event":"val", "task":"lm", "kind": kind, "seq": args.seq,
                     "val_loss": vavg, "val_ppl": vppl, "loss": vavg, "ppl": vppl})

    # write logs
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.logdir) / f"lm_{kind}_T{args.seq}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"metrics.jsonl", "a", encoding="utf-8") as f:
        for r in logs:
            f.write(json.dumps(r) + "\n")
    print("logs →", str(out_dir))

    # Save checkpoint (state_dict) if requested
    if args.save_ckpt:
        ckpt_path = Path(args.save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(ckpt_path))
        meta = {
            "saved_at": stamp,
            "hf_model": args.hf_model,
            "seq": args.seq,
            "steps": args.steps,
            "kd": args.kd,
            "kd_alpha": args.kd_alpha,
            "kd_tau": args.kd_tau,
            "freeze_non_attn": args.freeze_non_attn,
            "dtype": args.dtype,
            "hidden_mse": args.hidden_mse,
            "hidden_mse_weight": args.hidden_mse_weight,
            "spec_smooth": args.spec_smooth,
            "spec_smooth_weight": args.spec_smooth_weight,
        }
        with open(ckpt_path.with_suffix(".json"), "w", encoding="utf-8") as jf:
            json.dump(meta, jf, indent=2)
        print(f"[save] checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
