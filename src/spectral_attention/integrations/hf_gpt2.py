"""
Hugging Face GPT-2 integration: swap GPT2Attention for SpectralAttention.

Usage:
    from transformers import GPT2Config
    from spectral_attention.integrations import convert_gpt2_to_spectral

    cfg = GPT2Config(n_layer=2, n_head=4, n_embd=256)
    model = convert_gpt2_to_spectral(cfg)
    out = model(input_ids=torch.randint(0, cfg.vocab_size, (2, 64)))
"""

from dataclasses import dataclass
from typing import Optional

import torch

try:  # optional dependency
    from transformers.models.gpt2.modeling_gpt2 import (
        GPT2Attention,
        GPT2Model,
        GPT2Config,
        GPT2LMHeadModel,
    )
except Exception:  # pragma: no cover - allow import without transformers installed
    GPT2Attention = None  # type: ignore[assignment]
    GPT2Model = None      # type: ignore[assignment]
    GPT2Config = None     # type: ignore[assignment]
    GPT2LMHeadModel = None  # type: ignore[assignment]

from ..spectral_attention import SpectralAttention


@dataclass
class SpectralGPT2Options:
    use_dct: bool = False
    token_gate: bool = False
    dropout: float = 0.0


class SpectralGPT2Attention(GPT2Attention):  # type: ignore[misc]
    def __init__(self, config, *, opts: SpectralGPT2Options):
        super().__init__(config)
        self.spectral = SpectralAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            use_dct=opts.use_dct,
            token_gate=opts.token_gate,
            dropout=opts.dropout,
        )

    def forward(self, hidden_states, **kwargs):
        # GPT2Block expects two outputs to unpack: (attn_output, attn_weights)
        # Provide attn_output from SpectralAttention and dummy None for weights.
        attn_output = self.spectral(hidden_states)
        return (attn_output, None)


def convert_gpt2_to_spectral(config, *, opts: Optional[SpectralGPT2Options] = None, from_pretrained: Optional[str] = None):
    """Create a GPT-2 model where attention is replaced by SpectralAttention.

    If from_pretrained is provided, weights are loaded before patching.
    """
    if GPT2Model is None:
        raise RuntimeError("transformers not available; please install transformers to use this integration.")

    opts = opts or SpectralGPT2Options()
    if from_pretrained:
        model = GPT2Model.from_pretrained(from_pretrained, config=config)
    else:
        model = GPT2Model(config)

    # Monkey-patch all blocks
    for block in model.h:
        block.attn = SpectralGPT2Attention(config, opts=opts)
    return model


def convert_gpt2lm_to_spectral(config, *, opts: Optional[SpectralGPT2Options] = None, from_pretrained: Optional[str] = None):
    """Create a GPT-2 LMHead model where attention is replaced by SpectralAttention.

    This keeps embeddings and LM head; only the attention op is swapped.
    """
    if GPT2LMHeadModel is None:
        raise RuntimeError("transformers not available; please install transformers to use this integration.")

    opts = opts or SpectralGPT2Options()
    if from_pretrained:
        model = GPT2LMHeadModel.from_pretrained(from_pretrained, config=config)
    else:
        model = GPT2LMHeadModel(config)

    # Patch transformer blocks
    for block in model.transformer.h:
        block.attn = SpectralGPT2Attention(config, opts=opts)
    return model
