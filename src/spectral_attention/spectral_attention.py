# src/SpectralAttention/spectralAttention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- device + DCT helpers -----------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- DCT / IDCT fallback ----------
def dct_ii(x, dim=-1, norm="ortho"):
    """
    DCT-II fallback using FFT, works on CPU and GPU.
    """
    N = x.size(dim)
    # create extended signal
    x_ext = torch.cat([x, x.flip([dim])], dim=dim)
    X_fft = torch.fft.fft(x_ext, dim=dim)
    k = torch.arange(N, device=x.device)
    factor = torch.exp(-1j * torch.pi * k / (2 * N))
    result = (X_fft.index_select(dim, k) * factor).real
    if norm == "ortho":
        result[:, 0] /= torch.sqrt(torch.tensor(4*N, dtype=result.dtype, device=x.device))
        result[:, 1:] /= torch.sqrt(torch.tensor(2*N, dtype=result.dtype, device=x.device))
        result *= torch.sqrt(torch.tensor(2.0, dtype=result.dtype, device=x.device))
    return result

def idct_iii(X, dim=-1, norm="ortho"):
    """
    IDCT-III fallback (inverse of DCT-II).
    """
    N = X.size(dim)
    X_v = X.clone()
    if norm == "ortho":
        X_v[..., 0] /= torch.sqrt(torch.tensor(2.0, dtype=X.dtype, device=X.device))
    X_v = X_v * torch.exp(1j * torch.pi * torch.arange(N, device=X.device) / (2*N))
    X_full = torch.cat([X_v, torch.conj(X_v.flip([dim]))], dim=dim)
    x = torch.fft.ifft(X_full, dim=dim).real
    return x[..., :N]

# Detect native DCT
_HAS_TORCH_DCT = hasattr(torch.fft, "dct")


def _resolve_device(pref: str) -> torch.device:
    """
    'auto' -> use the input's device in forward
    'gpu'  -> cuda if available, else error
    'cpu'  -> cpu
    """
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Device preference 'gpu' was set but CUDA is not available.")
    # 'auto' is handled in forward via x.device; this is a placeholder
    return torch.device("cpu")


def _move_dim_to_last(x: torch.Tensor, dim: int) -> torch.Tensor:
    dim = dim if dim >= 0 else x.dim() + dim
    if dim == x.dim() - 1:
        return x
    perm = [d for d in range(x.dim()) if d != dim] + [dim]
    return x.permute(*perm)

def _move_last_to_dim(x: torch.Tensor, dim: int, orig_ndim: int) -> torch.Tensor:
    dim = dim if dim >= 0 else orig_ndim + dim
    if dim == orig_ndim - 1:
        return x
    perm = list(range(orig_ndim))
    last = orig_ndim - 1
    for i in range(orig_ndim - 1, dim, -1):
        perm[i] = i - 1
    perm[dim] = last
    return x.permute(*perm)


def dct2(x: torch.Tensor, dim: int = -1, norm: str | None = "ortho") -> torch.Tensor:
    """
    DCT-II via rFFT of even extension, matches SciPy dct(type=2, norm='ortho').
    Works on CPU/GPU. Safe for any 'dim' with broadcasting preserved.
    """
    orig_ndim = x.dim()
    x_last = _move_dim_to_last(x, dim)
    N = x_last.size(-1)

    # Even extension: [x, flip(x)]
    x_ext = torch.cat([x_last, x_last.flip(dims=[-1])], dim=-1)      # [..., 2N]
    X = torch.fft.rfft(x_ext, dim=-1)                                 # [..., N+1] complex

    # Phase and scaling
    k = torch.arange(N, device=x.device, dtype=x_last.real.dtype)     # [..., N]
    factor = torch.exp(-1j * math.pi * k / (2.0 * N))                 # complex
    coeff = (X[..., :N] * factor).real * 2.0                          # [..., N]

    if norm == "ortho":
        coeff[..., 0] = coeff[..., 0] / math.sqrt(2.0)
        coeff = coeff * math.sqrt(2.0 / N)

    return _move_last_to_dim(coeff, dim, orig_ndim)


def idct3(x: torch.Tensor, dim: int = -1, norm: str | None = "ortho") -> torch.Tensor:
    """
    IDCT-III that inverts dct2(..., norm='ortho').
    Rebuild an rFFT spectrum, irFFT length 2N, crop to first N.
    """
    orig_ndim = x.dim()
    x_last = _move_dim_to_last(x, dim)
    N = x_last.size(-1)

    # Undo orthonormal scaling
    if norm == "ortho":
        x0 = x_last[..., 0] / math.sqrt(2.0)
        xr = torch.cat([x0.unsqueeze(-1), x_last[..., 1:]], dim=-1) * math.sqrt(N / 2.0)
    else:
        xr = x_last

    k = torch.arange(N, device=x.device, dtype=x_last.real.dtype)
    factor = torch.exp(1j * math.pi * k / (2.0 * N))
    Xk = 0.5 * (xr * factor)                                          # [..., N] complex

    zeros = torch.zeros_like(Xk[..., :1])
    Xfull = torch.cat([Xk, zeros], dim=-1)                             # [..., N+1]
    x_ext = torch.fft.irfft(Xfull, n=2 * N, dim=-1).real               # [..., 2N]
    rec = x_ext[..., :N]

    return _move_last_to_dim(rec, dim, orig_ndim)


# -------------------------------- main module ----------------------------------

class SpectralAttention(nn.Module):
    """
    Spectral (FFT/DCT) global mixer: O(n log n).
    - rFFT/irFFT (complex) or DCT-II (real) path
    - per-head per-bin filters (gain + phase)
    - residual connection
    - optional token-gate (rank-1 in frequency)
    - device knob: 'auto' | 'cpu' | 'gpu'
    - optional FlashAttention integration for QK^T softmax steps
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        use_dct: bool = False,
        token_gate: bool = False,
        dropout: float = 0.0,
        device: str = "auto",  # 'auto' | 'cpu' | 'gpu'
        use_flash: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_dct = use_dct
        self.token_gate = token_gate
        self.use_flash = use_flash
        self.device_pref = (device or "auto").lower()
        self.has_torch_dct = hasattr(torch.fft, "dct") and hasattr(torch.fft, "idct")
        
        # Check FlashAttention availability
        self.has_flash_attn = False
        if use_flash:
            try:
                import flash_attn
                from flash_attn import flash_attn_func
                self.has_flash_attn = True
                self.flash_attn_func = flash_attn_func
            except ImportError:
                print("Warning: FlashAttention requested but not available. Falling back to torch.fft path.")
                self.has_flash_attn = False

        # Projections
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Spectral params (lazy-init based on seq length)
        self.register_buffer("_initialized_bins", torch.tensor(0), persistent=False)
        self.log_gain: nn.Parameter | None = None  # [H, n_bins]
        self.phase: nn.Parameter | None = None     # [H, n_bins]

        # Optional token-conditioned gate (rank-1 over frequency)
        if self.token_gate:
            self.alpha = nn.Linear(self.d_head, 1, bias=True)
            nn.init.zeros_(self.alpha.weight)
            nn.init.constant_(self.alpha.bias, 1e-2)  # was 0.0

    def _maybe_init_bins(self, T: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize per-head spectral bins based on seq length.

        Smart init: low-frequency emphasis + smooth decay + small phase noise.
        Can be disabled by setting environment variable SPECTRAL_NO_SMART_INIT=1
        (the training script may expose a CLI flag to toggle this environment var).
        """
        n_bins = T if self.use_dct else (T // 2 + 1)
        if int(self._initialized_bins.item()) == n_bins:
            return
        sp_dtype = torch.float32
        self.log_gain = nn.Parameter(torch.zeros(self.n_heads, n_bins, device=device, dtype=sp_dtype))
        self.phase    = nn.Parameter(torch.zeros(self.n_heads, n_bins, device=device, dtype=sp_dtype))

        # Smart init logic
        import os
        disable = os.environ.get("SPECTRAL_NO_SMART_INIT") == "1"
        if not disable:
            # Frequency axis 0..1
            freq = torch.linspace(0, 1, n_bins, device=device, dtype=sp_dtype)
            # Base shape: gentle negative slope plus mild curvature
            base = 0.05 - 0.25 * (freq ** 1.2)
            # Add a light Gaussian smoothing kernel via conv (simulate smoothing)
            if n_bins > 5:
                kernel = torch.tensor([0.05, 0.2, 0.5, 0.2, 0.05], device=device, dtype=sp_dtype)
                pad = (kernel.numel() - 1) // 2
                padded = torch.nn.functional.pad(base.unsqueeze(0).unsqueeze(0), (pad, pad), mode='reflect')
                smooth = torch.nn.functional.conv1d(padded, kernel.view(1,1,-1)).squeeze(0).squeeze(0)
                base = smooth[:n_bins]
            # Slight per-head variation
            head_noise = torch.randn(self.n_heads, 1, device=device, dtype=sp_dtype) * 0.01
            self.log_gain.data.copy_(base.unsqueeze(0) + head_noise)
            # Small phase noise (do not bias phase strongly)
            self.phase.data.normal_(mean=0.0, std=0.01)
        self._initialized_bins = torch.tensor(n_bins, device=device, dtype=torch.long)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, T, d_model] -> y: [B, T, d_model]
        """
        # Resolve device policy
        orig_device = x.device
        if self.device_pref == "auto":
            target = orig_device
        else:
            target = _resolve_device(self.device_pref)
            if x.device != target:
                x = x.to(target)
        if self.device_pref in ("cpu", "gpu") and next(self.parameters(), torch.tensor(0, device=target)).device != target:
            self.to(target)

        B, T, _ = x.shape
        device, in_dtype = x.device, x.dtype

        # Projections
        q, k, v = self.W_qkv(x).chunk(3, dim=-1)
        
        # If FlashAttention is available and enabled, use it for QK^T computation
        if self.use_flash and self.has_flash_attn:
            # Reshape for FlashAttention: [B, T, n_heads, d_head]
            q = q.view(B, T, self.n_heads, self.d_head)
            k = k.view(B, T, self.n_heads, self.d_head)
            v = v.view(B, T, self.n_heads, self.d_head)
            
            # Use FlashAttention for efficient QK^T computation
            # FlashAttention expects inputs in [B, T, n_heads, d_head] format
            try:
                attn_output = self.flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
                # attn_output shape: [B, T, n_heads, d_head]
                # Reshape to [B, n_heads, T, d_head] for spectral processing
                v = attn_output.transpose(1, 2)
            except Exception as e:
                print(f"FlashAttention failed, falling back to spectral mixing: {e}")
                # Fall back to spectral mixing without attention
                v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        else:
            # Use V for spectral mixing without FlashAttention
            v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, Dh]

        # Per-seq-length spectral params
        self._maybe_init_bins(T, device, torch.float32)
        assert self.log_gain is not None and self.phase is not None

        if self.use_dct:
            if self.has_torch_dct:
                v_t = torch.fft.dct(v.to(torch.float32), type=2, dim=2, norm="ortho")
            else:
                v_t = dct2(v.to(torch.float32), dim=2, norm="ortho")

            gain = torch.exp(self.log_gain)  # [H, T]
            v_t = v_t * gain.unsqueeze(0).unsqueeze(-1)

            if self.token_gate:
                g = torch.tanh(self.alpha(v.to(in_dtype))).squeeze(-1)  # [B,H,T]
                v_t = v_t + g.unsqueeze(-1)

            if self.has_torch_dct:
                y_h = torch.fft.idct(v_t, type=2, dim=2, norm="ortho").to(in_dtype)
            else:
                y_h = idct3(v_t, dim=2, norm="ortho").to(in_dtype)

        else:
            # rFFT path (complex)
            v_f = torch.fft.rfft(v.to(torch.float32), dim=2)           # [B,H,Fr,Dh]
            mag  = torch.exp(self.log_gain)                            # [H,Fr]
            comp = torch.polar(mag, self.phase)                        # [H,Fr] (complex)
            v_f = v_f * comp.unsqueeze(0).unsqueeze(-1)

            if self.token_gate:
                alpha = torch.tanh(self.alpha(v.to(in_dtype))).squeeze(-1)  # [B,H,T]
                alpha_f = torch.fft.rfft(alpha.to(torch.float32), dim=2)     # [B,H,Fr]
                v_f = v_f + alpha_f.unsqueeze(-1) * 0.05

            y_h = torch.fft.irfft(v_f, n=T, dim=2).to(in_dtype)        # [B,H,T,Dh]

        # Merge heads, project, residual
        y = y_h.transpose(1, 2).contiguous().view(B, T, self.d_model)
        y = self.drop(self.W_o(y))
        y = x + y

        # Keep output on original device for 'auto'
        if self.device_pref == "auto" and y.device != orig_device:
            y = y.to(orig_device)
        return y
