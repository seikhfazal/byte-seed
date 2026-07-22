<div align="center">

# ByteSeed

**A small decoder-only Transformer implemented directly in PyTorch for learning, experimentation, and local language-model development.**

[![CPU CI](https://github.com/seikhfazal/byte-seed/actions/workflows/ci.yml/badge.svg)](https://github.com/seikhfazal/byte-seed/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-dependency-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Overview](#overview) · [Install](#installation) · [Quick start](#quick-start) · [Training workflow](#training-workflow) · [Documentation](#documentation)

</div>

## Overview

ByteSeed is a compact GPT-style language-model project built from first principles with PyTorch. It is intended to make the essential pieces of a decoder-only Transformer approachable: tokenization, causal attention, training, supervised fine-tuning, checkpoint handling, provenance validation, and local interactive generation.

The project supports local CPU and CUDA execution, pretraining, supervised fine-tuning (SFT), evaluation scripts, and a terminal chat interface. It is deliberately small and experimental: it is a learning and local-development project, not a production language-model service.

## Highlights

- Decoder-only causal Transformer with manual multi-head causal self-attention by default and optional PyTorch SDPA execution.
- Optional inference-only KV caching for manual, SDPA, and auto-selected attention.
- Learned token and position embeddings, pre-norm residual blocks, LayerNorm, GELU MLPs, dropout, and tied input/output embeddings.
- SentencePiece BPE tokenization with required chat control tokens.
- Local pretraining, example-wise SFT, evaluation, generation benchmarking, and terminal chat.
- CPU and CUDA inference; automatic dtype selection uses fp16 on CUDA and fp32 on CPU.
- Versioned checkpoint kinds and deterministic, type-aware checkpoint selection.
- Exact pretraining resume for state-complete checkpoints with matching configuration and provenance.
- SHA-256 tokenizer and data-manifest identities for compatibility validation.
- Document-aware deterministic splitting, duplicate-group isolation, and exact evaluation-contamination guards for new data builds.
- Deterministic CPU-only tests and GitHub Actions CI.

## Model snapshot

### Architecture facts

| Property | Value |
| --- | --- |
| Default reference configuration | `ByteSeed-12M` (`configs/byteseed_12m.yaml`) |
| Architecture | Decoder-only, GPT-style Transformer |
| Context length | 256 tokens |
| Transformer blocks | 8 |
| Attention heads | 8 (40 dimensions per head) |
| Embedding width | 320 |
| MLP expansion | 4× embedding width with GELU |
| Dropout | 0.1 |
| Configured vocabulary capacity | 8,000 tokens |
| Framework | PyTorch |

The model ties the final language-model head to the token embedding table. The checked parameter-count utility reports **11,129,920 total/trainable parameters** for the current local `ByteSeed-12M` tokenizer snapshot, whose effective SentencePiece vocabulary is 3,699 tokens. The effective vocabulary—and therefore the exact parameter count—depends on the tokenizer artifact used at runtime; model configuration is aligned to that tokenizer when loaded.

### Checkpoint-specific facts

The current stable local checkpoint named by the repository is `checkpoints/anchor_v2_3_finetuned.pt`. Checkpoints and tokenizer binaries are intentionally ignored by Git, so a fresh clone does not include a runnable model artifact. See [Training Notes](docs/TRAINING_NOTES.md) and [Repository Hygiene](docs/REPO_HYGIENE.md) for the artifact policy.

### Configurable values

The repository also includes `ByteSeed-5M` and `ByteSeed-20M` YAML configurations. Architecture, training, and path settings are defined in `configs/`; the active tokenizer determines the effective vocabulary size.

## Architecture

ByteSeed predicts the next token at every position. Token IDs are embedded, combined with learned positional embeddings, passed through repeated pre-norm Transformer blocks, normalized once more, and projected through the tied language-model head.

```mermaid
flowchart LR
    A[Source documents] --> B[Data-quality audit]
    B --> C[Deterministic document split]
    C --> D[SentencePiece tokenization]
    D --> E[train.npy and val.npy]
    E --> F[Decoder-only Transformer]
    F --> G[Pretraining]
    G --> H[Optional supervised fine-tuning]
    H --> I[Local chat]
```

Each Transformer block applies LayerNorm, causal self-attention, a residual connection, LayerNorm, a GELU MLP, and a second residual connection. The default manual path uses an explicit lower-triangular mask; the optional SDPA path uses PyTorch's causal attention contract. Select `--attention-backend sdpa` (or `auto`) on supported PyTorch/device/dtype combinations. The backend changes execution only: parameters, state-dict keys, and checkpoint weights are shared. PyTorch chooses any optimized internal kernel, so no universal speedup or specific-kernel claim is made. See [Architecture](docs/ARCHITECTURE.md) for the concise implementation overview.

## Installation

ByteSeed requires Python 3.11 or later. The project is developed with Windows PowerShell commands, and the package/development installation used by CI is portable.

### Windows PowerShell

```powershell
git clone https://github.com/seikhfazal/byte-seed.git
cd byte-seed
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . -r requirements-dev.txt
```

### POSIX shells

```bash
git clone https://github.com/seikhfazal/byte-seed.git
cd byte-seed
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e . -r requirements-dev.txt
```

The base runtime dependencies are declared in [pyproject.toml](pyproject.toml); `requirements-dev.txt` adds the test dependency.

## Quick start

`chat.py` starts the local terminal interface with the `ByteSeed-12M` configuration and the `precise` preset. It selects the first available preferred local checkpoint, with `anchor_v2_3_finetuned.pt` first in that order. Interactive chat defaults to `--attention-backend auto`: it uses PyTorch SDPA when available and otherwise uses manual attention. Use `--attention-backend manual` to select the reference path explicitly; training and the global model/config default remain manual.

```powershell
python chat.py
```

Plain chat remains uncached. Add --kv-cache to reuse each layer's projected keys and values during one generation request:

~~~powershell
python chat.py --kv-cache
~~~

The optional cache is inference-only, works with manual and SDPA attention, and is discarded after every reply. It does not change model weights or checkpoint data. With learned absolute positions, ByteSeed invalidates the cache and uses the existing uncached path once the active context window must slide. The current inference wrapper rejects --compile together with --kv-cache.

To choose the stable local checkpoint explicitly:

```powershell
python chat.py --checkpoint checkpoints\anchor_v2_3_finetuned.pt
```

Useful verified options include `--preset balanced`, `--dtype fp32`, and `--dtype auto`. Chat starts stateless by default; use `/help` to view commands, including `/reset`, `/history on`, `/history off`, `/temp`, `/topk`, `/max`, `/raw`, and `/exit`.

Chat requires a local SentencePiece model and a compatible checkpoint. Neither is distributed through the Git repository by default.

## Training workflow

Training and data commands create ignored local artifacts. Review the linked guides and use a small, reviewed corpus before launching longer runs.

```powershell
# Inspect local data, then train the tokenizer.
python scripts/inspect_dataset.py
python -m src.byteseed.train_tokenizer --config configs/byteseed_12m.yaml

# Build document-aware train/validation arrays and provenance records.
python -m src.byteseed.prepare_data --config configs/byteseed_12m.yaml

# Pretrain using the configured schedule.
python -m src.byteseed.pretrain --config configs/byteseed_12m.yaml

# Run the targeted Anchor v2.3 SFT wrapper.
python scripts/run_anchor_v2_3_sft.py --config configs/byteseed_12m.yaml
```

For an exact automatic pretraining resume, use a state-complete checkpoint with matching tokenizer and data identities:

```powershell
python -m byteseed.pretrain --config configs/byteseed_12m.yaml --resume
```

Evaluation and benchmark scripts require local model artifacts. The historical retention check is:

```powershell
python scripts/eval_stable_v0_2.py --checkpoint checkpoints\anchor_v2_3_finetuned.pt
```

Detailed procedures and safety boundaries live in [Evaluation](docs/EVALUATION.md), [Training Notes](docs/TRAINING_NOTES.md), [Dataset Guide](docs/DATASET_GUIDE.md), and [Data Quality](docs/DATA_QUALITY.md).

## Reproducibility and checkpoint safety

Checkpoint schema version 1 distinguishes `pretrain`, `sft`, and `model_only` checkpoints. Selection is deterministic and type-aware: automatic exact pretraining resume accepts only compatible, state-complete pretraining checkpoints and does not silently downgrade to an inexact continuation.

New pretraining checkpoints record tokenizer identity, data-manifest identity, a complete optimizer continuation state, Python and PyTorch CPU RNG state, initialized CUDA RNG state when CUDA is active, AMP GradScaler state, early-stopping state, and a training-critical configuration snapshot. Tokenizer identity fingerprints model bytes, vocabulary size, and special-token IDs; data manifests fingerprint the token arrays and preprocessing identity.

The exact-resume guarantee is intentionally bounded. With matching supported software and hardware conditions, deterministic operations, matching training-critical configuration, and matching tokenizer/data manifests, resume restores the next stochastic and optimizer state. It does not promise bitwise-identical results across different hardware, PyTorch/CUDA versions, or nondeterministic kernels. Legacy and explicitly inexact continuations remain opt-in and are documented in [Training Notes](docs/TRAINING_NOTES.md).

## Data-quality policy

New pretraining-data builds preserve document boundaries before tokenization. Canonical duplicate groups are assigned as a unit, so duplicate content cannot cross the train/validation boundary. Splitting is deterministic from the duplicate-group fingerprint, configured seed, and validation ratio.

All registered evaluation suites are checked for exact normalized overlap in
document text and structured fields. The generalization SFT builder additionally
checks near wording against all three suites. New reports record the exact suite
versions and ordered prompt IDs audited. Contaminated builds fail by default;
the historical override is explicit and recorded in the quality report and
manifest identity.

**Anchor-retention regression: 9/9.**

**Held-out generalization: not yet measured.**

The nine Anchor prompts occur verbatim in historical Anchor v2.3 SFT material. The retention result is not a generalization, accuracy, or benchmark claim. The candidate paraphrase suite remains unverified until exact audit, manifest, and checkpoint identities prove it clean. See [Evaluation](docs/EVALUATION.md) and [Data Quality](docs/DATA_QUALITY.md) for the full policy.

## Testing

Run the complete deterministic suite from the repository root:

```powershell
python -m pytest -q
```

To mirror the source compilation check used by CI:

```powershell
python -m compileall -q src scripts chat.py tests
```

GitHub Actions runs CPU-only tests on Ubuntu with Python 3.11. The suite uses small synthetic models and temporary artifacts; it covers manual/SDPA attention parity, KV-cache prefill/decode/rollover parity, causal behavior, model shapes, generation, tokenizer handling, datasets and SFT masking, checkpoint schema and selection, exact resume, provenance, document splitting, evaluation contamination, RNG isolation, and evaluation/benchmark report validation. It does not require local checkpoints, tokenizer binaries, CUDA, or network access. See [Testing](docs/TESTING.md).

## Repository structure

```text
.
├── src/byteseed/       # Model, tokenizer, data, training, chat, checkpoint, and provenance code
├── configs/            # 5M, 12M, and 20M YAML configurations
├── scripts/            # Inspection, evaluation, benchmark, SFT, and utility entry points
├── tests/              # Deterministic CPU test suite
├── docs/               # Architecture, training, data-quality, testing, and safety notes
├── data/               # Small tracked examples and data guidance; generated/imported data is ignored
├── tokenizer/          # Local SentencePiece artifacts; binary model/vocabulary files are ignored
├── checkpoints/        # Local model checkpoints; ignored by Git
├── chat.py             # Root terminal-chat launcher
└── pyproject.toml      # Package metadata and runtime dependencies
```

## Current limitations

- ByteSeed is a small model trained on small local, synthetic, and curated datasets; it is not comparable to modern large-scale language models.
- The current checkpoint is best treated as a local, single-turn assistant experiment. Multi-turn history is off by default because the training examples are mostly single-turn.
- Outputs can hallucinate, repeat, or confuse concepts outside the Anchor training material. Do not use them for factual, medical, legal, financial, security-critical, or other high-stakes decisions.
- The historical Anchor score is a retention regression with known training overlap; held-out generalization has not been measured.
- KV caching is optional and off by default. It avoids repeated key/value projection only while the active learned-position context fits within block_size; rollover uses uncached generation for correctness. Actual benefit depends on device, dtype, sequence length, backend, and workload.
- There is no distributed-training workflow or broad benchmark suite.

## Roadmap

Repository audits identify the following next areas of work:

- A provenance-verified candidate paraphrase run and broader held-out evaluation coverage.
- Expanded clean, reviewed training corpora and stronger SFT data quality checks.
- Broader environment-aware benchmark methodology without cross-system superiority claims.
- Environment-specific cached/uncached measurements without universal performance claims.
- Packaging and public-artifact polish, including a documented policy for checkpoint and tokenizer distribution.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [How to run chat](docs/HOW_TO_RUN.md)
- [Training notes and resume contract](docs/TRAINING_NOTES.md)
- [Dataset guide](docs/DATASET_GUIDE.md)
- [Data-quality and provenance policy](docs/DATA_QUALITY.md)
- [Evaluation suites and report schemas](docs/EVALUATION.md)
- [Generalization SFT v1 data and holdout](docs/GENERALIZATION_SFT_V1.md)
- [Testing and CI](docs/TESTING.md)
- [Known limitations](docs/LIMITATIONS.md)
- [Data handling guidance](data/README_DATA.md)
- [Repository hygiene](docs/REPO_HYGIENE.md)

## Contributing

Contributions are most useful when they are focused and verifiable:

1. Create a focused branch and keep the change scoped.
2. Update or add deterministic tests for behavioral changes.
3. Run `python -m pytest -q` before proposing the change.
4. Do not commit private data, checkpoints, tokenizer binaries, processed datasets, generated outputs, or secrets.

The repository’s data and artifact rules are documented in [Repository Hygiene](docs/REPO_HYGIENE.md) and [Data README](data/README_DATA.md).

## License

ByteSeed is released under the [MIT License](LICENSE).
