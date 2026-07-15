from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .checkpoint import CheckpointOperation, discover_checkpoint


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
    """Return the latest model-bearing checkpoint for legacy callers."""
    checkpoint = discover_checkpoint(checkpoint_dir, CheckpointOperation.MODEL_LOAD)
    return checkpoint.path if checkpoint is not None else None


def warn_if_tiny_dataset(num_tokens: int, block_size: int) -> None:
    if num_tokens < block_size * 20:
        print(
            "Warning: this dataset is very small. The model may memorize it quickly. "
            "Add more clean Markdown notes for better results."
        )

