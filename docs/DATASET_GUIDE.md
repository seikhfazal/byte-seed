# ByteSeed Dataset Guide

## Why Assistant Behavior Needs Examples

A tiny model learns from repeated patterns. If you want ByteSeed to act like a personal assistant, the dataset needs examples of assistant behavior: asking for logs, giving safe commands, explaining study topics, and avoiding fake confidence.

Markdown notes help with broad style and facts. Chat examples help with response shape.

## ByteSeed-12M Is Limited

ByteSeed-12M has about 12.5M parameters. That is tiny compared with production assistants. It can learn tone, short routines, and simple local workflow patterns, but it will not become ChatGPT just by adding more data.

Expect repetition, mistakes, weak reasoning, and overfitting if the dataset is small.

## Pretraining Markdown vs Chat Fine-Tuning

Pretraining on Markdown teaches general text patterns and project knowledge. It is useful for notes, explanations, and vocabulary.

Fine-tuning on chat examples teaches the model how to answer user messages in a consistent format.

For ByteSeed, use both:

1. Markdown corpus for identity, project notes, and explanations.
2. JSONL SFT examples for direct assistant behavior.

## Why Random Internet Scraping Is Avoided

Random scraping can introduce copyrighted text, low-quality data, private information, toxic content, and license problems. ByteSeed should use reviewed local notes and small licensed public subsets only when needed.

## Recommended ByteSeed Dataset Path

1. Write personal Markdown notes; each source file remains one document.
2. Add assistant SFT examples.
3. Optionally import a small public dataset subset.
4. Run a short training test.
5. Start full training only after validation.

New data preparation assigns document and duplicate groups before tokenization,
rejects registered evaluation-prompt contamination by default, and writes a
deterministic quality report. The old combined corpus is retained only for
historical workflows. See [DATA_QUALITY.md](DATA_QUALITY.md).

## Practical Workflow

```powershell
python scripts/inspect_dataset.py
python scripts/build_personal_dataset.py
python -m src.byteseed.train_tokenizer --config configs/byteseed_12m.yaml
python -m src.byteseed.prepare_data --config configs/byteseed_12m.yaml
python -m src.byteseed.pretrain --config configs/byteseed_12m.yaml --max-iters 100
```

If CUDA runs out of memory on the RTX 4050 6GB GPU, reduce `batch_size` first, then reduce `block_size` if needed.
## Optional Public Datasets Later

Do not download public datasets automatically. Treat these as future options only, and review license terms before importing any content.

- `databricks/databricks-dolly-15k`: useful instruction data. License is CC BY-SA 3.0, so attribution and share-alike requirements apply.
- `OpenAssistant/oasst1`: assistant conversation tree data. It needs a converter, filtering, and cleaning before it fits ByteSeed JSONL format.
- `TinyStories`: useful for tiny model fluency and simple language. It is not assistant behavior data, so it should not replace local SFT examples.

Do not include dataset content in this guide. Do not auto-download public datasets from scripts unless that workflow is explicitly requested and reviewed later.
