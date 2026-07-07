from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    paths = sorted(Path(checkpoint_dir).glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def warn_if_tiny_dataset(num_tokens: int, block_size: int) -> None:
    if num_tokens < block_size * 20:
        print(
            "Warning: this dataset is very small. The model may memorize it quickly. "
            "Add more clean Markdown notes for better results."
        )

