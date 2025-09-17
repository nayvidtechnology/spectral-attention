import torch

def device_of(x: torch.Tensor) -> torch.device:
    return x.device

def choose_device(pref: str = "auto") -> torch.device:
    p = (pref or "auto").lower()
    if p == "cpu":  return torch.device("cpu")
    if p == "gpu":  return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # auto
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def resolve_device(pref: str = "auto") -> torch.device:
    """
    Resolve device from a user flag with strict semantics:
    - "gpu": require CUDA; exit if unavailable
    - "cpu": force CPU
    - "auto": CUDA if available, else CPU
    """
    p = (pref or "auto").lower()
    if p == "cpu":
        return torch.device("cpu")
    if p == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("--device gpu requested but CUDA is not available on this machine")
        return torch.device("cuda")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def parse_amp_dtype(name: str | None):
    """Map a string to a torch dtype for autocast. Returns None if disabled."""
    if not name: return None
    key = name.lower()
    if key in ("bf16", "bfloat16"): return torch.bfloat16
    if key in ("fp16", "float16"):   return torch.float16
    if key in ("fp32", "float32"):   return torch.float32
    if key in ("none", "off", "disable"): return None
    # default: None
    return None

def has_native_dct() -> bool:
    return hasattr(torch.fft, "dct") and hasattr(torch.fft, "idct")

def set_fast_matmul():
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
