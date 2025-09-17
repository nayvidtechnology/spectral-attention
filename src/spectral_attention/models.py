# src/spectral_attention/models.py
import math
import torch
import torch.nn as nn

from .blocks import SpectralEncoder
from .vanilla_blocks import VanillaEncoder  # you already added this earlier

class GPTPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos.weight, std=0.01)

    def forward(self, B, T, device):
        idx = torch.arange(T, device=device)
        return self.pos(idx)[None, :, :].expand(B, T, -1)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, ids):
        return self.emb(ids)

class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.out = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, h):
        return self.out(h)

class SpectralLM(nn.Module):
    def __init__(self, vocab_size: int, d_model=512, n_heads=8, depth=6, max_len=4096, dropout=0.1, use_dct=False):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model)
        self.pos = GPTPositionalEmbedding(d_model, max_len)
        self.enc = SpectralEncoder(depth=depth, d_model=d_model, n_heads=n_heads, use_dct=use_dct, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = LMHead(d_model, vocab_size)

    def forward(self, ids):  # ids: [B,T]
        B, T = ids.shape
        x = self.tok(ids) + self.pos(B, T, ids.device)
        h = self.enc(x)
        h = self.norm(h)
        return self.head(h)  # [B,T,V]

class VanillaLM(nn.Module):
    def __init__(self, vocab_size: int, d_model=512, n_heads=8, depth=6, max_len=4096, dropout=0.1):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model)
        self.pos = GPTPositionalEmbedding(d_model, max_len)
        self.enc = VanillaEncoder(depth=depth, d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = LMHead(d_model, vocab_size)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok(ids) + self.pos(B, T, ids.device)
        h = self.enc(x)
        h = self.norm(h)
        return self.head(h)
