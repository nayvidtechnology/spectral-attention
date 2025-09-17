# src/spectral_attention/blocks.py
import torch
import torch.nn as nn
from .spectral_attention import SpectralAttention



class SpectralEncoderBlock(nn.Module):
    """
    Residual encoder block:
      x -> LayerNorm -> SpectralAttention -> +x -> LayerNorm -> MLP -> +res
    Keeps things simple and stable; drop-in for Transformer encoder layers.
    """
    def __init__(self, d_model=512, n_heads=8, use_dct=False, dropout=0.1, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = SpectralAttention(d_model, n_heads, use_dct=use_dct, token_gate=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(mlp_ratio * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm -> spectral mixer
        x = self.attn(self.norm1(x))
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class SpectralEncoder(nn.Module):
    """
    Stack of SpectralEncoderBlock(s) with an input/output LayerNorm.
    """
    def __init__(self, d_model=512, n_heads=8, depth=6, use_dct=False, dropout=0.1, mlp_ratio=4):
        super().__init__()
        self.in_norm  = nn.LayerNorm(d_model)
        self.blocks   = nn.ModuleList([
            SpectralEncoderBlock(d_model, n_heads, use_dct=use_dct, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out_norm(x)


class TokenEmbedding(nn.Module):
    """
    Simple token embedding + (optional) learned positional embedding.
    For continuous inputs, just drop the embedding and feed features directly.
    """
    def __init__(self, vocab_size, d_model, max_len=8192, use_pos_emb=True):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.use_pos = use_pos_emb
        if use_pos_emb:
            self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x_ids):
        """
        x_ids: [B, T] integer token ids
        """
        B, T = x_ids.shape
        h = self.tok(x_ids)
        if self.use_pos:
            pos = torch.arange(T, device=x_ids.device).unsqueeze(0)
            h = h + self.pos(pos)
        return h


class SpectralEncoderModel(nn.Module):
    """
    Minimal language-model-style wrapper:
      ids -> embed -> spectral encoder -> LM head
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, depth=6, use_dct=False, dropout=0.1, mlp_ratio=4):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.encoder = SpectralEncoder(d_model, n_heads, depth, use_dct, dropout, mlp_ratio)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x_ids):
        """
        x_ids: [B, T]  -> logits: [B, T, vocab_size]
        """
        h = self.embed(x_ids)
        h = self.encoder(h)
        return self.lm_head(h)
