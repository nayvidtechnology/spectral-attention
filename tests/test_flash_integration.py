# tests/test_flash_integration.py
import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectral_attention import SpectralAttention


def dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("T,d_model,heads", [
    (256, 256, 8),
    (1024, 512, 8),
])
@pytest.mark.parametrize("use_dct", [False, True])
@pytest.mark.parametrize("use_flash", [False, True])
def test_flash_integration_shape_and_dtype(T, d_model, heads, use_dct, use_flash):
    """Test that FlashAttention integration preserves shape and dtype."""
    if use_dct and not hasattr(torch.fft, "dct"):
        pytest.skip("torch.fft.dct not available in this PyTorch build")
    
    device = dev()
    x = torch.randn(2, T, d_model, device=device, dtype=torch.float16)
    
    blk = SpectralAttention(
        d_model, heads, 
        use_dct=use_dct, 
        use_flash=use_flash, 
        dropout=0.0
    ).to(device)
    
    y = blk(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_flash_fallback_consistency():
    """Test that FlashAttention fallback produces consistent results."""
    device = dev()
    d_model, heads, T = 256, 8, 512
    
    # Create input
    x = torch.randn(2, T, d_model, device=device, dtype=torch.float16)
    
    # Test with FlashAttention enabled (will fallback if not available)
    blk_flash = SpectralAttention(d_model, heads, use_flash=True, dropout=0.0).to(device)
    
    # Test without FlashAttention  
    blk_no_flash = SpectralAttention(d_model, heads, use_flash=False, dropout=0.0).to(device)
    
    # Copy parameters to ensure identical weights
    blk_flash.load_state_dict(blk_no_flash.state_dict())
    
    with torch.no_grad():
        y_flash = blk_flash(x)
        y_no_flash = blk_no_flash(x)
    
    # Results should have same shape and dtype
    assert y_flash.shape == y_no_flash.shape
    assert y_flash.dtype == y_no_flash.dtype
    assert torch.isfinite(y_flash).all()
    assert torch.isfinite(y_no_flash).all()


def test_flash_flag_initialization():
    """Test that use_flash flag is properly handled during initialization."""
    d_model, heads = 256, 8
    
    # Test with FlashAttention enabled
    blk_flash = SpectralAttention(d_model, heads, use_flash=True)
    assert blk_flash.use_flash == True
    
    # Test without FlashAttention
    blk_no_flash = SpectralAttention(d_model, heads, use_flash=False)
    assert blk_no_flash.use_flash == False
    
    # Test default (should be False)
    blk_default = SpectralAttention(d_model, heads)
    assert blk_default.use_flash == False


def test_flash_with_token_gate():
    """Test FlashAttention works with token gate enabled."""
    device = dev()
    d_model, heads, T = 256, 8, 512
    
    x = torch.randn(2, T, d_model, device=device, dtype=torch.float16)
    
    blk = SpectralAttention(
        d_model, heads, 
        use_flash=True, 
        token_gate=True, 
        dropout=0.0
    ).to(device)
    
    y = blk(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    # Run basic test manually if called directly
    test_flash_flag_initialization()
    test_flash_integration_shape_and_dtype(256, 256, 8, False, True)
    test_flash_fallback_consistency()
    print("All FlashAttention integration tests passed!")