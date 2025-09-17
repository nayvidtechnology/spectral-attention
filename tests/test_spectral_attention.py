# tests/test_spectral_attention.py
import torch
import pytest

from spectral_attention import SpectralAttention


def dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("T,d_model,heads", [
    (128, 256, 8),
    (1024, 512, 8),
    (1536, 768, 12),
])
@pytest.mark.parametrize("use_dct", [False, True])
@pytest.mark.parametrize("token_gate", [False, True])
def test_forward_shape_and_dtype(T, d_model, heads, use_dct, token_gate):
    if use_dct and not hasattr(torch.fft, "dct"):
        pytest.skip("torch.fft.dct not available in this PyTorch build")
    device = dev()
    x = torch.randn(2, T, d_model, device=device)
    blk = SpectralAttention(d_model, heads, use_dct=use_dct, token_gate=token_gate, dropout=0.0).to(device)
    y = blk(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_backward_no_nan_amp():
    device = dev()
    torch.manual_seed(0)
    B, T, d_model, H = 2, 1024, 512, 8
    x = torch.randn(B, T, d_model, device=device)
    blk = SpectralAttention(d_model, H, use_dct=False, token_gate=False, dropout=0.1).to(device)

    #scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    opt = torch.optim.AdamW(blk.parameters(), lr=1e-3)

    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        #with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            y = blk(x)
            loss = (y ** 2).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for p in blk.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Found non-finite gradient"
        scaler.step(opt)
        scaler.update()

    assert torch.isfinite(y).all()


def test_bins_update_when_seq_changes():
    device = dev()
    d_model, H = 256, 8
    blk = SpectralAttention(d_model, H, use_dct=False).to(device)

    x1 = torch.randn(1, 257, d_model, device=device)
    _ = blk(x1)
    bins1 = int(blk._initialized_bins.item())
    assert bins1 == (x1.shape[1] // 2 + 1)

    x2 = torch.randn(1, 512, d_model, device=device)
    _ = blk(x2)
    bins2 = int(blk._initialized_bins.item())
    assert bins2 == (x2.shape[1] // 2 + 1)
    assert bins2 != bins1


def test_dct_mode_real_and_stable():
    if not hasattr(torch.fft, "dct"):
        pytest.skip("torch.fft.dct not available in this PyTorch build")
    device = dev()
    d_model, H = 128, 4
    blk = SpectralAttention(d_model, H, use_dct=True).to(device)
    x = torch.randn(3, 333, d_model, device=device)
    y = blk(x)
    assert torch.isfinite(y).all()
    assert y.dtype == x.dtype
    assert y.shape == x.shape


def test_token_gate_toggle_effect():
    device = dev()
    d_model, H, T = 256, 8, 512
    x = torch.randn(2, T, d_model, device=device)

    a = SpectralAttention(d_model, H, use_dct=False, token_gate=False).to(device)
    b = SpectralAttention(d_model, H, use_dct=False, token_gate=True).to(device)
    # copy non-gate params so only the gate differs
    with torch.no_grad():
        b.W_qkv.load_state_dict(a.W_qkv.state_dict())
        b.W_o.load_state_dict(a.W_o.state_dict())
        if a.log_gain is not None and b.log_gain is not None:
            b.log_gain.copy_(a.log_gain)
        if a.phase is not None and b.phase is not None:
            b.phase.copy_(a.phase)

    ya = a(x)
    yb = b(x)
    # Not necessarily huge difference, but should not be identical everywhere
    assert not torch.allclose(ya, yb)
