from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data_quality import read_pretraining_documents


def read_markdown_corpus(raw_data_dir: str | Path) -> str:
    """Join source documents for tokenizer training without losing prep boundaries."""

    return "\n\n".join(
        document.text.strip() for document in read_pretraining_documents(raw_data_dir)
    )


def read_legacy_markdown_corpus(raw_data_dir: str | Path) -> str:
    """Reproduce the historical top-level Markdown concatenation explicitly."""

    raw_path = Path(raw_data_dir)
    files = sorted(raw_path.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {raw_path}.")
    return "\n\n".join(
        text
        for text in (
            path.read_text(encoding="utf-8-sig").strip() for path in files
        )
        if text
    )


class TokenDataset:
    def __init__(self, data: np.ndarray, block_size: int, device: str):
        if len(data) <= block_size:
            raise ValueError("Processed data is too short for the configured block_size.")
        self.data = torch.from_numpy(data.astype(np.int64))
        self.block_size = block_size
        self.device = device

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        num_starts = len(self.data) - self.block_size
        ix = torch.randint(num_starts, (batch_size,))
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)


def _repeat_if_tiny(data: np.ndarray, block_size: int, split: str) -> np.ndarray:
    needed = block_size + 2
    if len(data) >= needed:
        return data
    repeats = needed // max(1, len(data)) + 1
    print(f"Warning: {split} split has only {len(data)} tokens; repeating it for a runnable tiny-data demo.")
    return np.tile(data, repeats)


def load_processed(processed_data_dir: str | Path, block_size: int, device: str) -> tuple[TokenDataset, TokenDataset]:
    processed = Path(processed_data_dir)
    train_path = processed / "train.npy"
    val_path = processed / "val.npy"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Processed files missing in {processed}. Run prepare_data.py before training."
        )
    return (
        TokenDataset(np.load(train_path), block_size, device),
        TokenDataset(np.load(val_path), block_size, device),
    )


