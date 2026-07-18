# JARVIS — an LLM built from scratch (plus a real 1B brain)

A small GPT-style language model, written in plain PyTorch and trained
entirely on this Mac — plus an optional genuinely-smart brain: Llama 3.2 1B
running locally through Ollama (free, offline, no API keys).

## Quick start

```bash
python3 jarvis.py        # the full assistant: best brain + internet tools
python3 jarvis.py --voice          # replies spoken aloud (macOS)
python3 jarvis.py --brain tiny     # force your from-scratch model
python3 chat.py                    # pure from-scratch model, no tools

# rebuild the from-scratch model yourself:
python3 gen_data.py      # 1. create the training data (data/jarvis.txt)
python3 train.py         # 2. train the small model (~13 min on M1)
python3 chat.py          # 3. talk to it
```

## The three brains

| Brain | Parameters | What it is |
|---|---|---|
| `checkpoints/jarvis.pt` | 1.8M | Trained from scratch here, in ~13 min |
| `checkpoints/jarvis-xl.pt` | ~10M | Same, scaled up (use `chat.py --ckpt` to compare) |
| Llama 3.2 1B via Ollama | 1.24B | Open-weight model by Meta, runs locally in `bin/ollama` |

`jarvis.py` picks Llama automatically when available; facts still come from
the live tools (clock, wttr.in weather, BBC news, Wikipedia, local math) and
every reply is tagged with which brain answered.

## What's in here

| File | What it does |
|---|---|
| `model.py` | The transformer itself — embeddings, multi-head causal self-attention, feed-forward blocks. This is the same architecture family as ChatGPT/Claude, just tiny (~1.8M parameters vs their hundreds of billions). |
| `gen_data.py` | Synthesizes ~1.2 MB of `USER:`/`JARVIS:` dialogues from templates. This is the model's entire universe — it knows only what's in this file. |
| `train.py` | The training loop: sample random chunks of text, predict the next character, measure the error (loss), nudge the weights, repeat thousands of times. |
| `chat.py` | Loads the trained weights and runs an interactive chat loop. |

## How it actually works (the short version)

1. **Tokenization** — text is split into units the model can handle. Big
   LLMs use word-pieces; we use single characters to keep it simple.
   Our vocabulary is just ~67 characters.
2. **The only task: predict the next character.** Given `"USER: hello\nJARVIS: Good d"`,
   the model learns that `a` probably comes next. That's the whole trick —
   chat behavior falls out of next-token prediction on dialogue-formatted text.
3. **Attention** is what makes it a transformer: every character looks back
   at all the characters before it and learns *which ones matter* for
   predicting what comes next.
4. **Generation** — to reply, we feed in the conversation so far and sample
   one character at a time, feeding each one back in. Temperature controls
   how adventurous the sampling is.

## Honest expectations

This model knows nothing outside its 1.2 MB of training text. It will
answer in Jarvis style, make up plausible-looking numbers, and sometimes
mash sentences together — which, at a small scale, is exactly what
"hallucination" is. That's the point: you can see the mechanics of a real
LLM without the scale that hides them.

## Ways to level up

- **More training**: `python3 train.py --iters 8000` — lower loss, cleaner replies.
- **Bigger model**: raise `n_layer`, `n_embd`, `block_size` in `model.py`
  (watch your RAM; this Mac has 8 GB).
- **Your own data**: point `train.py --data` at any text file — your notes,
  a book, code — and it will learn to write like that instead.
- **Real tokenizer**: swap characters for BPE word-pieces (see minbpe/nanoGPT
  for reference implementations).
- **A genuinely smart local Jarvis**: training frontier-quality models takes
  data centers, but you can *run* open-weight models (Llama, Mistral, Qwen)
  locally and offline with [Ollama](https://ollama.com) — no API, free.
  Pair one with this project's chat loop and persona for the best of both.
