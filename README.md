# ByteSeed

ByteSeed is a tiny GPT-style decoder-only Transformer built from scratch in PyTorch for learning how LLMs work.

## Status

Current stable baby assistant checkpoint: `anchor_v2_3`.

Anchor v2.3 is a tiny targeted patch for underfitting wording and CUDA false troubleshooting confusion. It is not a broad model upgrade.

v0.3-speed adds inference dtype options and better benchmarking. It does not change training or model architecture.

The current stable local checkpoint is:

```text
checkpoints/anchor_v2_3_finetuned.pt
```

`python chat.py` auto-selects this checkpoint when it exists.

Suggested GitHub topics: pytorch, transformer, gpt, llm, language-model, from-scratch, machine-learning, deep-learning, cuda, sentencepiece

## Features

- Manually implemented GPT-style model
- Tokenizer pipeline with SentencePiece
- Local training loop and inference/generation
- CUDA support
- Chat CLI with `python chat.py`
- Stateless single-turn mode by default
- Checkpoint saving, loading, and auto-selection
- Supervised fine-tuning workflow
- Dataset preparation and cleaning scripts
- Evaluation scripts
- Generation benchmark script

## What ByteSeed Is Not

- Not a ChatGPT replacement
- Not trained on internet-scale data
- Not a general-purpose assistant yet
- Not based on Hugging Face Transformers

## Hardware Used

This project has been developed and tested on:

- Windows 11
- Python 3.11
- NVIDIA RTX 4050 Laptop GPU with 6GB VRAM
- 16GB RAM

CPU execution is useful for smoke tests, but training is expected to be slow without CUDA.

## Model Size

Current default model:

- `ByteSeed-12M`
- Around 11.1M parameters

The local parameter count observed during development was about `11,129,920`.

## How To Run

From Windows PowerShell:

```powershell
cd D:\ByteSeed
.\.venv\Scripts\activate
python chat.py
```

To explicitly select the stable checkpoint:

```powershell
python chat.py --checkpoint checkpoints\anchor_v2_3_finetuned.pt
```

Useful launch examples:

```powershell
python chat.py
python chat.py --dtype fp32
python chat.py --preset balanced
```

Useful chat commands:

- `/reset`
- `/history`
- `/history on`
- `/history off`
- `/temp <value>`
- `/topk <value>`
- `/max <value>`
- `/raw`
- `/help`
- `/exit`

Default history mode is off because the current assistant works best as a single-turn model.

See [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) for more details.

## How To Evaluate And Benchmark

Count parameters:

```powershell
python scripts/count_params.py --config configs/byteseed_12m.yaml
```

Compile-check source files:

```powershell
python -m compileall src scripts chat.py
```

Run the stable regression evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\eval_stable_v0_2.py --checkpoint checkpoints\anchor_v2_3_finetuned.pt
```

Benchmark dtype examples:

```powershell
python scripts/benchmark_generation.py --dtype fp32
python scripts/benchmark_generation.py --dtype fp16
python scripts/benchmark_generation.py --dtype auto
```

`torch.compile` is optional and experimental. Use `--compile` only when you want to test startup overhead and local compatibility.

Training notes are documented in [docs/TRAINING_NOTES.md](docs/TRAINING_NOTES.md).

## Demo Transcript

```text
python chat.py

User: who are you?
ByteSeed: I am ByteSeed, a small local assistant for DSA study and ByteSeed workflow. I keep answers short.

User: what is a stack ?
ByteSeed: A stack is LIFO: last in, first out. Push and pop removes the top.

User: What is overfitting?
ByteSeed: Overfitting means a model memorizes training data and performs worse on validation or new unseen data.

User: How do I run ByteSeed chat?
ByteSeed: cd D:/ByteSeed, then run python chat.py.

User: Should I upload checkpoints to GitHub?
ByteSeed: Do not commit checkpoints by default. Keep them in checkpoints/ locally or use external storage.
```

These are early local outputs from a tiny model and may vary depending on sampling settings.

See [docs/DEMO_TRANSCRIPT.md](docs/DEMO_TRANSCRIPT.md).

## Limitations

ByteSeed is a small educational model. It can still hallucinate, confuse concepts outside its anchor training data, and produce unreliable answers. It is not safe for factual, medical, legal, financial, or security-critical use.

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Repo Hygiene And Checkpoints

Do not commit checkpoints by default. Keep generated checkpoint files in `checkpoints/` locally or move them to external storage later if needed.

Before making the repository public, run:

```powershell
git status
git status --ignored
```

Confirm that `.venv/`, `checkpoints/`, tokenizer binary files, processed data, runs, logs, and secrets are not staged.

See [docs/REPO_HYGIENE.md](docs/REPO_HYGIENE.md) and [docs/PUBLIC_RELEASE_CHECKLIST.md](docs/PUBLIC_RELEASE_CHECKLIST.md).
