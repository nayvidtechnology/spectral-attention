import torch
from spectral_attention import SpectralAttention

x = torch.randn(2, 1024, 512)
m = SpectralAttention(512, 8, use_dct=False)
y = m(x)
print("ok:", y.shape)


