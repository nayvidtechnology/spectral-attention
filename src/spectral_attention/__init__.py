# src/spectralAttention/__init__.py

"""
Spectral Attention
------------------
Frequency-domain alternative to self-attention using rFFT/DCT
with per-frequency learnable filters.

Exports:
    - SpectralAttention
    - SpectralEncoderBlock
    - SpectralEncoder
    - SpectralEncoderModel
"""

from .spectral_attention import SpectralAttention
from .blocks import SpectralEncoderBlock, SpectralEncoder, SpectralEncoderModel
# Integration helpers
try:
    from .integrations import convert_gpt2_to_spectral  # type: ignore
    from .integrations.hf_gpt2 import convert_gpt2lm_to_spectral  # type: ignore
except Exception:
    convert_gpt2_to_spectral = None
    convert_gpt2lm_to_spectral = None

__all__ = [
    "SpectralAttention",
    "SpectralEncoderBlock",
    "SpectralEncoder",
    "SpectralEncoderModel",
    "convert_gpt2_to_spectral",
    "convert_gpt2lm_to_spectral",
]

