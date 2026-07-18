"""
Trains the GPT on data/jarvis.txt, from random weights, on your Mac's GPU.

Run:  python3 train.py            (about 5 minutes on an M1)
      python3 train.py --iters 6000   for a better model, if you have time

Saves the model to checkpoints/jarvis.pt
"""
import argparse
import math
import time

import torch

from model import GPT, GPTConfig


def get_device():
    if torch.backends.mps.is_available():
        return "mps"        # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/jarvis.txt")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="checkpoints/jarvis.pt")
    # model size knobs (defaults = the small 1.8M model)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=192)
    p.add_argument("--block-size", type=int, default=192)
    args = p.parse_args()

    torch.manual_seed(1337)
    device = get_device()

    # ---- load text and build the character-level tokenizer
    with open(args.data) as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = int(0.95 * len(data))
    train_data, val_data = data[:n], data[n:]

    config = GPTConfig(vocab_size=len(chars), n_layer=args.n_layer,
                       n_head=args.n_head, n_embd=args.n_embd,
                       block_size=args.block_size)
    model = GPT(config).to(device)
    print("device: %s | data: %.2f MB | vocab: %d chars | model: %.2fM params"
          % (device, len(text) / 1e6, len(chars), model.num_params() / 1e6))

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - config.block_size - 1, (args.batch_size,))
        x = torch.stack([d[i:i + config.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + config.block_size] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}
        for split in ("train", "val"):
            vals = [model(*get_batch(split))[1].item() for _ in range(20)]
            losses[split] = sum(vals) / len(vals)
        model.train()
        return losses

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(it):
        warmup = 100
        if it < warmup:
            return args.lr * (it + 1) / warmup
        progress = (it - warmup) / max(1, args.iters - warmup)
        return 0.1 * args.lr + 0.45 * args.lr * (1 + math.cos(math.pi * progress))

    print("training for %d iterations..." % args.iters)
    t0 = time.time()
    for it in range(args.iters):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(it)
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % 250 == 0 or it == args.iters - 1:
            losses = estimate_loss()
            print("iter %5d | train loss %.3f | val loss %.3f | %.0fs elapsed"
                  % (it, losses["train"], losses["val"], time.time() - t0))

    torch.save({
        "model": model.state_dict(),
        "config": config.__dict__,
        "stoi": stoi,
        "itos": itos,
    }, args.out)
    print("saved model to %s" % args.out)

    # ---- show a raw sample so we can see what it learned
    prompt = "USER: hello jarvis\nJARVIS:"
    idx = torch.tensor([[stoi[c] for c in prompt]], device=device)
    out, _ = model.generate(idx, max_new_tokens=120)
    print("\n--- sample ---")
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
