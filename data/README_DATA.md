# ByteSeed Data README

ByteSeed uses small, local, plain-text datasets. Keep the data clean, legal, and safe to publish.

## data/raw/personal_assistant/

Put personal assistant Markdown notes here. These files teach ByteSeed repeated behavior patterns: identity, tone, study routines, coding help, troubleshooting steps, project memory, and boundaries.

Use generic examples. Do not include private notes, passwords, access tokens, personal emails, addresses, or secrets.

## data/raw/assistant_sft/

Put small supervised chat examples here as JSONL. Each line should look like:

```json
{"user":"question here","assistant":"answer here"}
```

These examples teach assistant-style responses more directly than Markdown pretraining text.

## data/public_imports/

This folder is for optional public dataset imports created by `scripts/import_public_assistant_data.py`. Imported files are ignored by Git by default.

Review licenses and content before using public data. Do not blindly commit imported datasets.

## What Not To Commit

Do not commit:

- Private notes
- Passwords or tokens
- Checkpoints
- Processed `.npy` data
- Large public dataset exports
- `.parquet` or `.arrow` files
- Anything from a source you do not have rights to use

## Adding Your Own Markdown Notes Safely

1. Create a small `.md` file under `data/raw/personal_assistant/` or `data/raw/`.
2. Use headings and short examples.
3. Remove personal details and secrets.
4. Run dataset inspection.
5. Run document-aware data preparation.

## Inspect The Dataset

```powershell
python scripts/inspect_dataset.py
```

## Optional Historical Combined Dataset

```powershell
python scripts/build_personal_dataset.py
```

This historical builder creates:

- `data/raw/byteseed_personal_assistant_corpus.md`
- `examples/byteseed_personal_assistant_sft.jsonl`

New pretraining preparation does not use the combined Markdown output. It reads
the original Markdown files under `data/raw/`, `data/raw/personal_assistant/`,
and `data/raw/generated/markdown/` as separate documents. Top-level JSONL
document records may contain either `text` or non-empty `user` and `assistant`
fields. Each record remains one document.

## Later Tokenizer And Training Data Steps

```powershell
python -m src.byteseed.train_tokenizer --config configs/byteseed_12m.yaml
python -m src.byteseed.prepare_data --config configs/byteseed_12m.yaml
```

Preparation deduplicates canonical-equivalent documents, assigns duplicate
groups deterministically before tokenization, checks train/validation leakage,
and rejects registered evaluation-prompt overlap by default. Historical
reproduction of known contaminated material requires the explicit
`--allow-eval-contamination` flag and is recorded as such.

See [docs/DATA_QUALITY.md](../docs/DATA_QUALITY.md) for the exact policy and
manifest compatibility rules.

Start with short training tests before long training.
