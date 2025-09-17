# src/spectral_attention/models_cls.py
import torch
import torch.nn as nn

from .blocks import SpectralEncoder
from .vanilla_blocks import VanillaEncoder


class CLSHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, D], assume token 0 == [CLS]
        cls = h[:, 0, :]
        return self.fc(self.dropout(self.norm(cls)))


class SpectralClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        depth: int = 6,
        max_len: int = 4096,
        dropout: float = 0.1,
        use_dct: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos.weight, std=0.01)
        self.enc = SpectralEncoder(d_model=d_model, n_heads=n_heads, depth=depth,
                                   use_dct=use_dct, dropout=dropout)
        self.head = CLSHead(d_model, num_classes, dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device)
        x = self.tok(input_ids) + self.pos(pos)[None, :, :]
        h = self.enc(x)  # classification: full-context mixing, no causal mask
        logits = self.head(h)
        return logits


class VanillaClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        depth: int = 6,
        max_len: int = 4096,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos.weight, std=0.01)
        self.enc = VanillaEncoder(depth=depth, d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.head = CLSHead(d_model, num_classes, dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device)
        x = self.tok(input_ids) + self.pos(pos)[None, :, :]
        # Vanilla encoder supports attention kwargs; for classification we pass None
        h = self.enc(x, attn_mask=None)
        logits = self.head(h)
        return logits
