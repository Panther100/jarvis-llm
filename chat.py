"""
Chat with your trained JARVIS.

Run:  python3 chat.py
      python3 chat.py --voice     to have replies spoken aloud (macOS)

Type your message and press enter. Ctrl+C or 'bye' to quit... though
JARVIS will happily say goodbye and keep listening anyway.
"""
import argparse
import subprocess
import sys

import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/jarvis.pt")
    p.add_argument("--voice", action="store_true",
                   help="speak replies aloud using macOS 'say'")
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    def encode(s):
        return [stoi[c] for c in s if c in stoi]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    print("JARVIS online. %.2fM parameters, trained locally. (ctrl+c to exit)\n"
          % (model.num_params() / 1e6))

    history = ""
    while True:
        try:
            user = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJARVIS: Powering down, sir.")
            break
        if not user:
            continue

        history += "USER: %s\nJARVIS:" % user.lower()
        # condition on the most recent context the model can see
        context = history[-config.block_size:]
        idx = torch.tensor([encode(context)], device=device)

        # stop generating once the model starts a new "USER:" line
        def stop_check(new_ids):
            tail = decode(new_ids[-8:])
            return "\nUSER" in tail or tail.endswith("\n\n")

        _, new_ids = model.generate(
            idx, max_new_tokens=220,
            temperature=args.temperature, top_k=40,
            stop_check=stop_check,
        )
        reply = decode(new_ids)
        for cut in ("\nUSER", "\n\n"):
            if cut in reply:
                reply = reply.split(cut)[0]
        reply = reply.strip()

        print("JARVIS: %s\n" % reply)
        history += " %s\n" % reply

        if args.voice and reply:
            # Daniel is the built-in British voice on macOS
            subprocess.run(["say", "-v", "Daniel", reply],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
