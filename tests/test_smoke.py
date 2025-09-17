import torch
from spectral_attention import SpectralAttention

def test_forward_shape_and_grad():
    x = torch.randn(2, 1024, 512)
    m = SpectralAttention(512, 8, use_dct=False)
    y = m(x)
    assert y.shape == x.shape
    loss = (y ** 2).mean()
    loss.backward()  # no NaNs in grads
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

def test_dynamic_seq_bins():
    m = SpectralAttention(512, 8, use_dct=False)
    for T in (257, 512, 1536):
        x = torch.randn(1, T, 512)
        y = m(x)
        assert y.shape == x.shape
