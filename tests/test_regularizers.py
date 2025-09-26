import torch
import pytest
try:
    from transformers import GPT2Config
    from spectral_attention import convert_gpt2lm_to_spectral
    _xf_available = True
except Exception as e:  # broad to catch hub version mismatches
    _xf_available = False
    _xf_err = e

@pytest.mark.parametrize("use_dct", [False, True])
@pytest.mark.parametrize("token_gate", [False, True])
def test_hidden_mse_zero_when_models_identical(use_dct, token_gate):
    if not _xf_available:
        pytest.skip(f"transformers unavailable: {_xf_err}")
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=64, n_ctx=64, vocab_size=100)
    student = convert_gpt2lm_to_spectral(cfg, from_pretrained=None, use_dct=use_dct, token_gate=token_gate)
    teacher = convert_gpt2lm_to_spectral(cfg, from_pretrained=None, use_dct=use_dct, token_gate=token_gate)
    # copy weights so they are identical
    teacher.load_state_dict(student.state_dict())
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    attn = (x != 0)
    with torch.no_grad():
        s_out = student(input_ids=x, attention_mask=attn)
        t_out = teacher(input_ids=x, attention_mask=attn)
    s_h = s_out.last_hidden_state
    t_h = t_out.last_hidden_state
    mse = torch.mean((s_h - t_h)**2).item()
    assert mse < 1e-8, f"Hidden MSE should be ~0 for identical models, got {mse}"


def finite_diff_l2(param: torch.Tensor) -> torch.Tensor:
    return (param[:,1:] - param[:,:-1]).pow(2).mean() if param.size(1) > 1 else torch.tensor(0.0)


def test_spectral_smoothness_penalty_decreases_with_smoothing():
    if not _xf_available:
        pytest.skip(f"transformers unavailable: {_xf_err}")
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=32, n_positions=64, n_ctx=64, vocab_size=100)
    model = convert_gpt2lm_to_spectral(cfg, from_pretrained=None)
    # Trigger lazy init by running forward once
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    attn = (x != 0)
    _ = model(input_ids=x, attention_mask=attn)
    penalties = []
    for m in model.modules():
        if hasattr(m, 'log_gain') and isinstance(getattr(m, 'log_gain'), torch.nn.Parameter) and m.log_gain is not None:
            penalties.append(finite_diff_l2(m.log_gain.detach()))
    assert penalties, "No spectral modules with log_gain found"
    baseline = torch.stack(penalties).mean().item()
    # Manually smooth log_gain
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, 'log_gain') and isinstance(getattr(m, 'log_gain'), torch.nn.Parameter) and m.log_gain is not None:
                m.log_gain[:] = torch.nn.functional.avg_pool1d(m.log_gain.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    penalties2 = []
    for m in model.modules():
        if hasattr(m, 'log_gain') and isinstance(getattr(m, 'log_gain'), torch.nn.Parameter) and m.log_gain is not None:
            penalties2.append(finite_diff_l2(m.log_gain.detach()))
    after = torch.stack(penalties2).mean().item()
    assert after <= baseline + 1e-6, f"Smoothness penalty should not increase after smoothing (baseline={baseline}, after={after})"
