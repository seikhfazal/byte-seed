from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import yaml

from .config import load_config
from .dataset import read_markdown_corpus
from .provenance import tokenizer_identity_from_processor
from .utils import ensure_dir

USER_DEFINED_SYMBOLS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


def train_tokenizer(config_path: str, vocab_size: int | None = None) -> None:
    cfg = load_config(config_path, {"vocab_size": vocab_size})
    tokenizer_dir = ensure_dir(cfg.tokenizer_dir)
    corpus = read_markdown_corpus(cfg.raw_data_dir)
    corpus_path = tokenizer_dir / "corpus.txt"
    corpus_path.write_text(corpus, encoding="utf-8")
    requested_vocab = cfg.vocab_size
    target_vocab = min(requested_vocab, max(1000, len(corpus) // 6))
    print(f"requested_vocab_size: {requested_vocab}")
    print(f"target_vocab_size: {target_vocab}")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(tokenizer_dir / "byteseed"),
        vocab_size=target_vocab,
        model_type="bpe",
        character_coverage=1.0,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        hard_vocab_limit=False,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
    )
    model_path = tokenizer_dir / "byteseed.model"
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    actual_vocab = processor.get_piece_size()
    meta = {
        "requested_vocab_size": requested_vocab,
        "target_vocab_size": target_vocab,
        "actual_vocab_size": actual_vocab,
        "user_defined_symbols": USER_DEFINED_SYMBOLS,
        "model_path": str(model_path),
        "corpus_path": str(corpus_path),
        "tokenizer_identity": tokenizer_identity_from_processor(model_path, processor),
    }
    meta_path = tokenizer_dir / "tokenizer_meta.yaml"
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    json_meta_path = tokenizer_dir / "tokenizer_meta.json"
    json_meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Tokenizer saved to {model_path}")
    print(f"actual_vocab_size: {actual_vocab}")
    print(f"Tokenizer metadata saved to {meta_path}")
    print(f"Tokenizer JSON metadata saved to {json_meta_path}")
    print("Special-token round-trip check:")
    for symbol in USER_DEFINED_SYMBOLS:
        ids = processor.encode(symbol, out_type=int)
        decoded = processor.decode(ids)
        survived = decoded == symbol
        print(f"  {symbol}: ids={ids} decoded={decoded!r} survived={survived}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--vocab-size", type=int, default=None)
    args = parser.parse_args()
    train_tokenizer(args.config, args.vocab_size)


if __name__ == "__main__":
    main()
