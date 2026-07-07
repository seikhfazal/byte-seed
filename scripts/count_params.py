from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from byteseed.config import align_config_to_tokenizer, load_config
from byteseed.model import GPT
from byteseed.tokenizer import ByteSeedTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configured_vocab_size = cfg.vocab_size
    try:
        tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    except FileNotFoundError:
        tokenizer = None
        effective_vocab_size = cfg.vocab_size
    else:
        align_config_to_tokenizer(cfg, tokenizer, verbose=False)
        effective_vocab_size = cfg.vocab_size
    model = GPT(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model_name: {cfg.model_name}")
    print(f"configured_vocab_size: {configured_vocab_size:,}")
    print(f"effective_vocab_size: {effective_vocab_size:,}")
    if tokenizer is None:
        print("tokenizer_status: missing, using configured vocab size")
    else:
        print("tokenizer_status: found, using tokenizer vocab size")
    print(f"total_parameters: {total:,}")
    print(f"trainable_parameters: {trainable:,}")


if __name__ == "__main__":
    main()
