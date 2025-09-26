# src/spectral_attention/layers/hybrid_layer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from ..spectral_attention import SpectralAttention
from ..holonomy import HolonomyAttention


class StandardAttention(nn.Module):
    """
    Standard attention mechanism to be compatible with HolonomyAttention interface.
    Takes Q, K, V tensors [B, H, T, D_head] and returns [B, H, T, D_head].
    """
    
    def __init__(self, d_head: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(d_head)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention.
        
        Args:
            Q: Query tensor [batch, heads, seq_len, dim]
            K: Key tensor [batch, heads, seq_len, dim] 
            V: Value tensor [batch, heads, seq_len, dim]
            mask: Optional causal mask [seq_len, seq_len] or [batch, heads, seq_len, seq_len]
            
        Returns:
            Output tensor [batch, heads, seq_len, dim]
        """
        B, H, T, D = Q.shape
        assert H == self.n_heads, f"Expected {self.n_heads} heads, got {H}"
        assert D == self.d_head, f"Expected {self.d_head} dim, got {D}"
        
        # Compute attention scores
        scores = torch.einsum('bhtd,bhsd->bhts', Q, K) * self.scale  # [B, H, T, T]
        
        # Apply causal mask if provided
        if mask is not None:
            if mask.dim() == 2:  # [T, T] - broadcast to all batches and heads
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
            elif mask.dim() == 3:  # Assume [B, T, T] - broadcast to all heads
                mask = mask.unsqueeze(1)  # [B, 1, T, T]
            # mask should now be [1, 1, T, T] or [B, 1, T, T] or [B, H, T, T]
            
            # Apply mask (where mask is False, set to large negative value)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, T, T]
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        out = torch.einsum('bhts,bhsd->bhtd', attn_weights, V)  # [B, H, T, D]
        
        return out


class SpectralAttentionWrapper(nn.Module):
    """
    Wrapper for SpectralAttention to work with Q, K, V interface.
    Note: SpectralAttention doesn't use K,V directly but expects full input x.
    """
    
    def __init__(self, d_head: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.d_model = d_head * n_heads
        self.spectral_attn = SpectralAttention(
            d_model=self.d_model, 
            n_heads=n_heads, 
            dropout=dropout
        )
        
    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass using SpectralAttention.
        
        Args:
            Q: Query tensor [batch, heads, seq_len, dim] - used as input
            K: Key tensor [batch, heads, seq_len, dim] - ignored
            V: Value tensor [batch, heads, seq_len, dim] - ignored
            mask: Optional causal mask
            
        Returns:
            Output tensor [batch, heads, seq_len, dim]
        """
        B, H, T, D = Q.shape
        
        # Reshape Q from [B, H, T, D] to [B, T, H*D] for SpectralAttention
        x = Q.transpose(1, 2).contiguous().view(B, T, H * D)
        
        # Apply spectral attention
        y = self.spectral_attn(x, mask=mask)  # [B, T, H*D]
        
        # Reshape back to [B, H, T, D]
        output = y.view(B, T, H, D).transpose(1, 2)
        
        return output


class HybridAttentionLayer(nn.Module):
    """
    Multi-head Hybrid Attention Layer where heads can be of different attention types.
    
    Supports mixing "standard", "spectral", and "holonomy" head types.
    Each head type processes a subset of the total heads, then outputs are
    concatenated and projected back to d_model.
    """
    
    def __init__(
        self,
        d_model: int,
        head_types: List[str],
        dropout: float = 0.0
    ):
        """
        Args:
            d_model: Model dimension
            head_types: List of head types, e.g., ["standard", "spectral", "holonomy", "standard"]
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.head_types = head_types
        self.n_heads = len(head_types)
        self.dropout = dropout
        
        assert d_model % self.n_heads == 0, f"d_model ({d_model}) must be divisible by number of heads ({self.n_heads})"
        self.d_head = d_model // self.n_heads
        
        # Input projection to Q, K, V
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Create attention modules for each head type
        self.attention_modules = nn.ModuleDict()
        self.head_type_counts = {}
        
        # Count heads of each type
        for head_type in set(head_types):
            count = head_types.count(head_type)
            self.head_type_counts[head_type] = count
            
            if head_type == "standard":
                self.attention_modules[head_type] = StandardAttention(
                    d_head=self.d_head, 
                    n_heads=count, 
                    dropout=dropout
                )
            elif head_type == "spectral":
                self.attention_modules[head_type] = SpectralAttentionWrapper(
                    d_head=self.d_head, 
                    n_heads=count, 
                    dropout=dropout
                )
            elif head_type == "holonomy":
                self.attention_modules[head_type] = HolonomyAttention(
                    d_head=self.d_head, 
                    n_heads=count, 
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown head type: {head_type}")
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through hybrid attention layer.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        B, T, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.W_qkv(x)  # [B, T, 3*d_model]
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head)  # [B, T, 3, H, D]
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        
        # Process heads by type
        outputs = []
        head_idx = 0
        
        for head_type in ["standard", "spectral", "holonomy"]:
            if head_type not in self.head_type_counts:
                continue
                
            count = self.head_type_counts[head_type]
            
            # Extract heads for this type
            q_heads = q[:, head_idx:head_idx + count]  # [B, count, T, D]
            k_heads = k[:, head_idx:head_idx + count]  # [B, count, T, D]
            v_heads = v[:, head_idx:head_idx + count]  # [B, count, T, D]
            
            # Apply attention
            attn_module = self.attention_modules[head_type]
            out_heads = attn_module(q_heads, k_heads, v_heads, mask=mask)  # [B, count, T, D]
            
            outputs.append(out_heads)
            head_idx += count
        
        # Concatenate outputs from all head types
        if outputs:
            concat_output = torch.cat(outputs, dim=1)  # [B, H, T, D]
        else:
            concat_output = torch.zeros(B, self.n_heads, T, self.d_head, device=x.device, dtype=x.dtype)
        
        # Reshape and project output
        concat_output = concat_output.transpose(1, 2).contiguous()  # [B, T, H, D]
        concat_output = concat_output.view(B, T, self.d_model)  # [B, T, d_model]
        
        # Final projection and residual connection
        output = self.W_o(concat_output)
        output = self.dropout_layer(output)
        output = x + output  # residual connection
        
        return output