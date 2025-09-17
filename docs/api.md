# API

## `spectral_attn.SpectralAttention(d_model, n_heads, use_dct=False, token_gate=False, dropout=0.0)`
Drop-in Attention replacement. Input shape `[B, T, d_model]` → same shape.

## `spectral_attn.SpectralEncoderBlock(d_model=512, n_heads=8, use_dct=False, dropout=0.1)`
Residual encoder block with SpectralAttention + MLP.
