from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


ATTENTION_BACKENDS = ("manual", "sdpa", "auto")


INT_FIELDS = {
    "vocab_size",
    "block_size",
    "n_layer",
    "n_head",
    "n_embd",
    "batch_size",
    "gradient_accumulation_steps",
    "max_iters",
    "eval_interval",
    "eval_iters",
    "warmup_iters",
    "seed",
    "early_stopping_patience",
}

FLOAT_FIELDS = {
    "dropout",
    "learning_rate",
    "weight_decay",
    "train_split",
}


@dataclass
class ByteSeedConfig:
    model_name: str = "ByteSeed-12M"
    vocab_size: int = 8000
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 320
    dropout: float = 0.1
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    weight_decay: float = 0.1
    warmup_iters: int = 200
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    tokenizer_dir: str = "tokenizer"
    checkpoint_dir: str = "checkpoints"
    train_split: float = 0.9
    seed: int = 1337
    device: str = "auto"
    early_stopping_patience: int = 0
    attention_backend: str = "manual"

    def __post_init__(self) -> None:
        for field in INT_FIELDS:
            setattr(self, field, _to_int(field, getattr(self, field)))
        for field in FLOAT_FIELDS:
            setattr(self, field, _to_float(field, getattr(self, field)))
        self.model_name = str(self.model_name)
        self.raw_data_dir = str(self.raw_data_dir)
        self.processed_data_dir = str(self.processed_data_dir)
        self.tokenizer_dir = str(self.tokenizer_dir)
        self.checkpoint_dir = str(self.checkpoint_dir)
        self.device = str(self.device)
        self.attention_backend = str(self.attention_backend).strip().lower()
        self.validate()

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def validate(self) -> None:
        if self.n_head <= 0:
            raise ValueError("Config error: n_head must be positive.")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"Config error: n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})."
            )
        if not 0 < self.train_split < 1:
            raise ValueError("Config error: train_split must be between 0 and 1, for example 0.9.")
        if self.block_size <= 0:
            raise ValueError("Config error: block_size must be positive.")
        if self.vocab_size <= 0:
            raise ValueError("Config error: vocab_size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Config error: batch_size must be positive.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Config error: gradient_accumulation_steps must be positive.")
        if self.attention_backend not in ATTENTION_BACKENDS:
            choices = ", ".join(ATTENTION_BACKENDS)
            raise ValueError(
                f"Config error: attention_backend must be one of {choices}; "
                f"got {self.attention_backend!r}."
            )


def _to_int(field: str, value: Any) -> int:
    try:
        return int(float(value)) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config error: {field} must be an integer-like value, got {value!r}.") from exc


def _to_float(field: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config error: {field} must be a number, got {value!r}.") from exc


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ByteSeedConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}. Try configs/byteseed_12m.yaml.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return ByteSeedConfig(**data)


def config_from_checkpoint(raw_config: dict[str, Any], fallback_device: str | None = None) -> ByteSeedConfig:
    data = dict(raw_config)
    if fallback_device is not None:
        data["device"] = fallback_device
    return ByteSeedConfig(**data)


def align_config_to_tokenizer(cfg: ByteSeedConfig, tokenizer: Any, *, verbose: bool = True) -> ByteSeedConfig:
    actual_vocab_size = int(tokenizer.vocab_size)
    if cfg.vocab_size != actual_vocab_size:
        if verbose:
            print(
                f"Tokenizer vocab size is {actual_vocab_size}; using it instead of configured vocab_size={cfg.vocab_size}."
            )
        cfg.vocab_size = actual_vocab_size
        cfg.validate()
    return cfg
