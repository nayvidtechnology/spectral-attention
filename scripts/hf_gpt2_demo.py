import torch
from transformers import GPT2Config
from spectral_attention import convert_gpt2_to_spectral


def main():
    cfg = GPT2Config(n_layer=2, n_head=4, n_positions=512, n_embd=256)
    model = convert_gpt2_to_spectral(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    out = model(input_ids=x)
    last = out.last_hidden_state
    print("OK", tuple(last.shape))


if __name__ == "__main__":
    main()
