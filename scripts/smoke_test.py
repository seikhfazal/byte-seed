from __future__ import annotations

import shutil
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from byteseed.config import ByteSeedConfig
from byteseed.dataset import TokenDataset, read_markdown_corpus
from byteseed.model import GPT
from byteseed.train_tokenizer import train_tokenizer
from byteseed.prepare_data import prepare_data
from byteseed.tokenizer import ByteSeedTokenizer
from byteseed.utils import ensure_dir, set_seed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "data" / "raw" / "sample.md"
    assert sample.exists(), "data/raw/sample.md is missing."

    smoke_dir = root / "runs" / "smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    ensure_dir(smoke_dir)
    cfg_path = smoke_dir / "smoke.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "model_name: ByteSeed-Smoke",
                "vocab_size: 256",
                "block_size: 32",
                "n_layer: 2",
                "n_head: 2",
                "n_embd: 64",
                "dropout: 0.1",
                "batch_size: 4",
                "gradient_accumulation_steps: 1",
                "learning_rate: 3e-4",
                "max_iters: 3",
                "eval_interval: 2",
                "eval_iters: 2",
                "weight_decay: 0.01",
                "warmup_iters: 1",
                "raw_data_dir: data/raw",
                "processed_data_dir: runs/smoke/processed",
                "tokenizer_dir: runs/smoke/tokenizer",
                "checkpoint_dir: runs/smoke/checkpoints",
                "train_split: 0.9",
                "seed: 1337",
                "device: cpu",
                "early_stopping_patience: 0",
            ]
        ),
        encoding="utf-8",
    )

    train_tokenizer(str(cfg_path), vocab_size=256)
    prepare_data(str(cfg_path))

    cfg = ByteSeedConfig(
        model_name="ByteSeed-Smoke",
        vocab_size=256,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        batch_size=4,
        device="cpu",
    )
    set_seed(cfg.seed)
    tokenizer = ByteSeedTokenizer(smoke_dir / "tokenizer")
    ids = tokenizer.encode(read_markdown_corpus(root / "data" / "raw"), add_bos=True, add_eos=True)
    data = TokenDataset(torch.tensor(ids).numpy(), cfg.block_size, "cpu")
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        xb, yb = data.get_batch(cfg.batch_size)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    prompt = torch.tensor([tokenizer.encode("Data structures", add_bos=True)], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=20, vocab_limit=tokenizer.vocab_size)
    print(tokenizer.decode(out[0].tolist()))
    print("ByteSeed smoke test passed.")


if __name__ == "__main__":
    main()


