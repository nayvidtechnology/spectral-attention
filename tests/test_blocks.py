# tests/test_blocks.py
import torch
import pytest

from spectral_attention import SpectralEncoderBlock, SpectralEncoder, SpectralEncoderModel


def dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("use_dct", [False, True])
def test_encoder_block_shapes(use_dct):
    if use_dct and not hasattr(torch.fft, "dct"):
        pytest.skip("torch.fft.dct not available in this PyTorch build")
    device = dev()
    B, T, D, H = 2, 256, 128, 4
    x = torch.randn(B, T, D, device=device)
    blk = SpectralEncoderBlock(d_model=D, n_heads=H, use_dct=use_dct, dropout=0.1).to(device)
    y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("use_dct", [False, True])
def test_encoder_stack(depth, use_dct):
    if use_dct and not hasattr(torch.fft, "dct"):
        pytest.skip("torch.fft.dct not available in this PyTorch build")
    device = dev()
    B, T, D, H = 2, 384, 192, 6
    x = torch.randn(B, T, D, device=device)
    enc = SpectralEncoder(d_model=D, n_heads=H, depth=depth, use_dct=use_dct, dropout=0.1).to(device)
    y = enc(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("use_dct", [False, True])
def test_lm_wrapper_logits(use_dct):
    if use_dct and not hasattr(torch.fft, "dct"):
        pytest.skip("torch.fft.dct not available in this PyTorch build")
    device = dev()
    B, T, V, D, H = 2, 128, 1000, 256, 8
    ids = torch.randint(0, V, (B, T), device=device)
    model = SpectralEncoderModel(vocab_size=V, d_model=D, n_heads=H, depth=2, use_dct=use_dct).to(device)
    logits = model(ids)
    assert logits.shape == (B, T, V)
    assert torch.isfinite(logits).all()

    # quick backward pass
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, V),
        torch.randint(0, V, (B * T,), device=device),
        reduction="mean",
    )
    loss.backward()
    # no NaNs in grads
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
