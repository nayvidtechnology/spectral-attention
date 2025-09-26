# src/spectral_attention/holonomy.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HolonomyAttention(nn.Module):
    """
    Holonomy Attention: applies holonomy-inspired rotation to Q before computing attention.
    
    Uses a learnable curvature matrix per head to rotate Q, then computes standard 
    scaled dot-product attention with optional causal masking.
    """
    
    def __init__(
        self,
        d_head: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(d_head)
        
        # Learnable curvature matrix per head for holonomy rotation
        # Each head gets its own rotation matrix [d_head, d_head]
        self.curvature = nn.Parameter(torch.randn(n_heads, d_head, d_head) * 0.02)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of Holonomy Attention.
        
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
        
        # Apply holonomy rotation to Q using learnable curvature matrix
        # Q_rot = Q @ curvature^T for each head
        Q_rot = torch.einsum('bhtd,hde->bhte', Q, self.curvature)  # [B, H, T, D]
        
        # Compute attention scores
        scores = torch.einsum('bhtd,bhsd->bhts', Q_rot, K) * self.scale  # [B, H, T, T]
        
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