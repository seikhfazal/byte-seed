# ByteSeed Project Memory

## Project Identity

ByteSeed is a tiny GPT-style language model grown from Markdown notes.

Default model: ByteSeed-12M.

Small model: ByteSeed-5M.

Experimental larger model: ByteSeed-20M.

Package name: `byteseed`.

CLI command name: `byteseed-chat`.

## Local Workflow

Common commands:

```powershell
python scripts/smoke_test.py
python scripts/count_params.py --config configs/byteseed_12m.yaml
python -m src.byteseed.train_tokenizer --config configs/byteseed_12m.yaml
python -m src.byteseed.prepare_data --config configs/byteseed_12m.yaml
python -m src.byteseed.pretrain --config configs/byteseed_12m.yaml --max-iters 100
```

## RTX 4050 Notes

The RTX 4050 Laptop GPU has limited VRAM for language model training. Prefer small batch sizes, gradient accumulation, mixed precision, and short validation tests.

If CUDA runs out of memory, reduce batch size first. Then reduce block size if needed.
