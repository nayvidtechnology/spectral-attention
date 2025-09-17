import json, argparse, torch, matplotlib.pyplot as plt
#from spectral_attn.spectral_attention import SpectralAttention
from spectral_attention.spectral_attention import SpectralAttention

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    args = p.parse_args()
    # Load your module state (customize as needed)
    state = torch.load(args.ckpt, map_location="cpu")
    # Expect to find 'log_gain' and 'phase' tensors in state
    log_gain = state["log_gain"] if "log_gain" in state else None
    phase    = state["phase"] if "phase" in state else None
    if log_gain is None or phase is None:
        raise RuntimeError("Checkpoint missing spectral params.")
    gain = torch.exp(log_gain).mean(0).numpy()
    ph   = phase.mean(0).numpy()
    fig, ax = plt.subplots(2, 1, figsize=(8,6), tight_layout=True)
    ax[0].plot(gain); ax[0].set_title("Mean Gain over rFFT bins")
    ax[1].plot(ph);   ax[1].set_title("Mean Phase over rFFT bins")
    plt.show()

if __name__ == "__main__":
    main()
