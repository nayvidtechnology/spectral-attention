import torch
import torch.nn as nn

class VanillaEncoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention
        y, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(y)
        x = self.norm1(x)
        # FFN
        y = self.ff(x)
        x = x + self.drop(y)
        x = self.norm2(x)
        return x

class VanillaEncoder(nn.Module):
    def __init__(self, depth=4, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([VanillaEncoderBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, **kw):
        for blk in self.layers:
            x = blk(x, **kw)
        return self.norm(x)
