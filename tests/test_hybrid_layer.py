# tests/test_hybrid_layer.py
import torch
import pytest
import math

from spectral_attention.layers.hybrid_layer import HybridAttentionLayer, StandardAttention, SpectralAttentionWrapper


def dev():
    """Get the best available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("batch_size,seq_len,d_model", [
    (2, 32, 512),
    (1, 64, 256), 
    (4, 128, 384),
])
def test_hybrid_layer_basic_forward(batch_size, seq_len, d_model):
    """Test basic forward pass with mixed head types."""
    device = dev()
    head_types = ["standard", "spectral", "holonomy", "standard"]
    
    layer = HybridAttentionLayer(d_model=d_model, head_types=head_types).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    output = layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert output.device == x.device
    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()


def test_hybrid_layer_acceptance_criteria():
    """Test the specific acceptance criteria: [4 standard, 2 spectral, 2 holonomy]."""
    device = dev()
    batch_size, seq_len, d_model = 2, 64, 512
    head_types = ["standard"] * 4 + ["spectral"] * 2 + ["holonomy"] * 2
    
    layer = HybridAttentionLayer(d_model=d_model, head_types=head_types).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    output = layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(output).all()
    
    # Check that the layer has the expected number of heads
    assert layer.n_heads == 8
    assert layer.head_type_counts["standard"] == 4
    assert layer.head_type_counts["spectral"] == 2
    assert layer.head_type_counts["holonomy"] == 2


def test_hybrid_layer_single_head_types():
    """Test with only one type of head."""
    device = dev()
    batch_size, seq_len, d_model = 2, 32, 256
    
    for head_type in ["standard", "spectral", "holonomy"]:
        head_types = [head_type] * 4
        layer = HybridAttentionLayer(d_model=d_model, head_types=head_types).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        output = layer(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert torch.isfinite(output).all()


def test_hybrid_layer_backward_pass():
    """Test that backward pass works without NaN gradients."""
    device = dev()
    batch_size, seq_len, d_model = 2, 32, 384  # 384 is divisible by 3
    head_types = ["standard", "spectral", "holonomy"]
    
    layer = HybridAttentionLayer(d_model=d_model, head_types=head_types).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    
    output = layer(x)
    loss = output.sum()
    loss.backward()
    
    # Check input gradients
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    
    # Check module parameters have gradients
    for param in layer.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_hybrid_layer_with_mask():
    """Test hybrid layer with attention mask."""
    device = dev()
    batch_size, seq_len, d_model = 2, 32, 256
    head_types = ["standard", "holonomy"]  # Skip spectral for simpler mask testing
    
    layer = HybridAttentionLayer(d_model=d_model, head_types=head_types).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    
    output = layer(x, mask=mask)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(output).all()


def test_hybrid_layer_different_configurations():
    """Test various head type configurations."""
    device = dev()
    batch_size, seq_len, d_model = 2, 32, 384
    
    configurations = [
        ["standard", "spectral"],
        ["holonomy", "holonomy", "standard"],
        ["spectral"] * 6,
        ["standard", "spectral", "holonomy", "spectral", "standard", "holonomy"],
    ]
    
    for head_types in configurations:
        layer = HybridAttentionLayer(d_model=d_model, head_types=head_types).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        output = layer(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert torch.isfinite(output).all()


def test_hybrid_layer_invalid_head_type():
    """Test that invalid head types raise an error."""
    with pytest.raises(ValueError, match="Unknown head type"):
        head_types = ["standard", "invalid_type"]
        HybridAttentionLayer(d_model=256, head_types=head_types)


def test_hybrid_layer_incompatible_dimensions():
    """Test that incompatible dimensions raise an error."""
    with pytest.raises(AssertionError, match="d_model.*must be divisible"):
        head_types = ["standard", "spectral", "holonomy"]  # 3 heads
        HybridAttentionLayer(d_model=256, head_types=head_types)  # 256 not divisible by 3


def test_standard_attention_component():
    """Test the StandardAttention component separately."""
    device = dev()
    batch_size, n_heads, seq_len, d_head = 2, 4, 32, 64
    
    attn = StandardAttention(d_head=d_head, n_heads=n_heads).to(device)
    
    Q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    V = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    
    output = attn(Q, K, V)
    
    assert output.shape == (batch_size, n_heads, seq_len, d_head)
    assert torch.isfinite(output).all()


def test_spectral_attention_wrapper():
    """Test the SpectralAttentionWrapper component separately."""
    device = dev()
    batch_size, n_heads, seq_len, d_head = 2, 4, 32, 64
    
    wrapper = SpectralAttentionWrapper(d_head=d_head, n_heads=n_heads).to(device)
    
    Q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    V = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    
    output = wrapper(Q, K, V)
    
    assert output.shape == (batch_size, n_heads, seq_len, d_head)
    assert torch.isfinite(output).all()


def test_hybrid_layer_output_deterministic():
    """Test that the same input produces the same output (deterministic)."""
    device = dev()
    batch_size, seq_len, d_model = 1, 16, 128
    head_types = ["standard", "holonomy"]
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    layer1 = HybridAttentionLayer(d_model=d_model, head_types=head_types).to(device)
    x1 = torch.randn(batch_size, seq_len, d_model, device=device)
    
    torch.manual_seed(42)
    layer2 = HybridAttentionLayer(d_model=d_model, head_types=head_types).to(device)
    x2 = x1.clone()
    
    # Set to eval mode to disable dropout
    layer1.eval()
    layer2.eval()
    
    with torch.no_grad():
        output1 = layer1(x1)
        output2 = layer2(x2)
    
    assert torch.allclose(output1, output2, atol=1e-6)