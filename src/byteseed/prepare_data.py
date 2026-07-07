from __future__ import annotations

import argparse

import numpy as np

from .config import load_config
from .dataset import read_markdown_corpus
from .tokenizer import ByteSeedTokenizer
from .utils import ensure_dir, set_seed, warn_if_tiny_dataset


def prepare_data(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    text = read_markdown_corpus(cfg.raw_data_dir)
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    split_idx = max(1, int(len(ids) * cfg.train_split))
    split_idx = min(split_idx, len(ids) - 1)
    processed_dir = ensure_dir(cfg.processed_data_dir)
    np.save(processed_dir / "train.npy", np.array(ids[:split_idx], dtype=np.uint16 if cfg.vocab_size <= 65535 else np.int32))
    np.save(processed_dir / "val.npy", np.array(ids[split_idx:], dtype=np.uint16 if cfg.vocab_size <= 65535 else np.int32))
    warn_if_tiny_dataset(len(ids), cfg.block_size)
    print(f"Saved {split_idx} train tokens and {len(ids) - split_idx} validation tokens to {processed_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    args = parser.parse_args()
    prepare_data(args.config)


if __name__ == "__main__":
    main()

