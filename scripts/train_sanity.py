import torch, torch.nn as nn
from spectral_attention import SpectralAttention

class TinyModel(nn.Module):
    def __init__(self, d=512, h=8):
        super().__init__()
        self.attn = SpectralAttention(d, h, use_dct=False, dropout=0.1)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x):
        x = self.attn(x)
        return x + self.ff(self.norm(x))

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B,T,D,H = 8,1024,512,8
    x = torch.randn(B,T,D, device=device)
    y_true = torch.randn(B,T,D, device=device)
    m = TinyModel(D,H).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    for step in range(50):
        opt.zero_grad(set_to_none=True)
        y = m(x)
        loss = (y - y_true).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (step+1) % 10 == 0:
            print(f"step {step+1:03d}  loss {loss.item():.4f}")
    # save plot-ready tensors
    torch.save({"log_gain": m.attn.log_gain.detach().cpu(),
                "phase":    m.attn.phase.detach().cpu()},
               r".\experiments\runs\2025-09-07T1730Z_seq4k_fft\artifacts\model_state.pt")
