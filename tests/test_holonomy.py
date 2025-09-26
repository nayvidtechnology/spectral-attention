# tests/test_holonomy.py
import torch
import pytest

from spectral_attention import HolonomyAttention


def dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("batch_size,n_heads,seq_len,d_head", [
    (2, 4, 32, 64),
    (1, 8, 64, 32), 
    (4, 6, 128, 48),
])
def test_forward_shape_and_dtype(batch_size, n_heads, seq_len, d_head):
    """Test that forward pass maintains correct shapes and dtypes."""
    device = dev()
    
    # Create random Q, K, V tensors
    Q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    V = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    
    # Create attention module
    attn = HolonomyAttention(d_head, n_heads, dropout=0.0).to(device)
    
    # Forward pass
    out = attn(Q, K, V)
    
    # Check output shape and dtype
    assert out.shape == (batch_size, n_heads, seq_len, d_head)
    assert out.dtype == Q.dtype
    assert torch.isfinite(out).all()


def test_causal_mask():
    """Test that causal masking works correctly."""
    device = dev()
    batch_size, n_heads, seq_len, d_head = 2, 4, 8, 16
    
    # Create random tensors
    Q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device) 
    V = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    
    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    
    attn = HolonomyAttention(d_head, n_heads, dropout=0.0).to(device)
    
    # Forward pass with mask
    out = attn(Q, K, V, mask=mask)
    
    assert out.shape == (batch_size, n_heads, seq_len, d_head)
    assert torch.isfinite(out).all()


def test_backward_pass():
    """Test that backward pass works without NaN gradients.""" 
    device = dev()
    batch_size, n_heads, seq_len, d_head = 2, 4, 16, 32
    
    Q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=True)
    K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=True)
    V = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, requires_grad=True)
    
    attn = HolonomyAttention(d_head, n_heads, dropout=0.1).to(device)
    
    # Forward pass
    out = attn(Q, K, V)
    loss = out.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist and are finite
    assert Q.grad is not None
    assert K.grad is not None  
    assert V.grad is not None
    assert torch.isfinite(Q.grad).all()
    assert torch.isfinite(K.grad).all()
    assert torch.isfinite(V.grad).all()
    
    # Check module parameters have gradients
    for param in attn.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_holonomy_rotation_effect():
    """Test that holonomy rotation actually changes the output compared to no rotation."""
    device = dev()
    batch_size, n_heads, seq_len, d_head = 1, 2, 8, 16
    
    Q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    V = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    
    # Holonomy attention with learned curvature
    attn_holonomy = HolonomyAttention(d_head, n_heads, dropout=0.0).to(device)
    
    # Create "identity" holonomy attention (curvature = identity matrix)
    attn_identity = HolonomyAttention(d_head, n_heads, dropout=0.0).to(device)
    with torch.no_grad():
        for h in range(n_heads):
            attn_identity.curvature.data[h] = torch.eye(d_head, device=device)
    
    out_holonomy = attn_holonomy(Q, K, V)
    out_identity = attn_identity(Q, K, V)
    
    # With random curvature vs identity, outputs should be different
    # (unless we get very unlucky with random initialization)
    assert not torch.allclose(out_holonomy, out_identity, atol=1e-6)


def test_different_mask_shapes():
    """Test different mask shapes work correctly."""
    device = dev()
    batch_size, n_heads, seq_len, d_head = 2, 3, 6, 8
    
    Q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    V = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    
    attn = HolonomyAttention(d_head, n_heads).to(device)
    
    # Test 2D mask [T, T]
    mask_2d = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    out_2d = attn(Q, K, V, mask=mask_2d)
    assert out_2d.shape == (batch_size, n_heads, seq_len, d_head)
    
    # Test 3D mask [B, T, T]  
    mask_3d = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=device, dtype=torch.bool))
    out_3d = attn(Q, K, V, mask=mask_3d)
    assert out_3d.shape == (batch_size, n_heads, seq_len, d_head)
    
    # Test 4D mask [B, H, T, T]
    mask_4d = torch.tril(torch.ones(batch_size, n_heads, seq_len, seq_len, device=device, dtype=torch.bool))
    out_4d = attn(Q, K, V, mask=mask_4d)
    assert out_4d.shape == (batch_size, n_heads, seq_len, d_head)