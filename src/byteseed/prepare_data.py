from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np

from .config import load_config
from .data_quality import (
    Document,
    build_data_quality_report,
    data_quality_preprocessing_identity,
    plan_document_dataset,
    read_pretraining_documents,
    write_data_quality_report,
)
from .provenance import build_pretraining_data_manifest, write_data_manifest
from .tokenizer import ByteSeedTokenizer
from .utils import ensure_dir, set_seed, warn_if_tiny_dataset


def encode_documents(
    documents: Iterable[Document], tokenizer: ByteSeedTokenizer
) -> list[int]:
    """Tokenize each complete document with its own BOS/EOS boundary."""

    token_ids: list[int] = []
    for document in documents:
        encoded = tokenizer.encode(document.text, add_bos=True, add_eos=True)
        if not encoded:
            raise ValueError(
                f"Document {document.document_id!r} produced no tokens."
            )
        token_ids.extend(encoded)
    return token_ids


def prepare_document_arrays(
    documents: Iterable[Document],
    tokenizer: ByteSeedTokenizer,
    *,
    train_split: float,
    split_seed: int,
    vocab_size: int,
    allow_eval_contamination: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Plan, split, and tokenize a synthetic or real document collection."""

    plan = plan_document_dataset(
        documents,
        seed=split_seed,
        validation_ratio=1.0 - float(train_split),
        allow_eval_contamination=allow_eval_contamination,
    )
    train_ids = encode_documents(plan.train_documents, tokenizer)
    validation_ids = encode_documents(plan.validation_documents, tokenizer)
    dtype = np.uint16 if vocab_size <= 65535 else np.int32
    report = build_data_quality_report(
        plan,
        train_token_count=len(train_ids),
        validation_token_count=len(validation_ids),
    )
    return (
        np.asarray(train_ids, dtype=dtype),
        np.asarray(validation_ids, dtype=dtype),
        report,
    )


def prepare_data(
    config_path: str, *, allow_eval_contamination: bool = False
) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    documents = read_pretraining_documents(cfg.raw_data_dir)
    train_ids, validation_ids, quality_report = prepare_document_arrays(
        documents,
        tokenizer,
        train_split=cfg.train_split,
        split_seed=cfg.seed,
        vocab_size=cfg.vocab_size,
        allow_eval_contamination=allow_eval_contamination,
    )
    processed_dir = ensure_dir(cfg.processed_data_dir)
    np.save(processed_dir / "train.npy", train_ids)
    np.save(processed_dir / "val.npy", validation_ids)
    write_data_quality_report(
        processed_dir / "data_quality_report.json", quality_report
    )
    manifest = build_pretraining_data_manifest(
        processed_dir,
        tokenizer_identity=tokenizer.identity,
        train_split=cfg.train_split,
        preprocessing_identity=data_quality_preprocessing_identity(quality_report),
    )
    write_data_manifest(processed_dir / "data_manifest.json", manifest)
    warn_if_tiny_dataset(len(train_ids) + len(validation_ids), cfg.block_size)
    counts = quality_report["counts"]
    print(
        f"Saved {len(train_ids)} train tokens and {len(validation_ids)} "
        f"validation tokens to {processed_dir}."
    )
    print(
        f"Data quality: {counts['accepted_unique_documents']} unique documents, "
        f"{counts['removed_duplicates']} duplicates removed, "
        f"{counts['contamination_matches']} evaluation-prompt matches."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument(
        "--allow-eval-contamination",
        action="store_true",
        help=(
            "Allow known evaluation overlap only for historical reproduction; "
            "the report records the override."
        ),
    )
    args = parser.parse_args()
    prepare_data(
        args.config,
        allow_eval_contamination=args.allow_eval_contamination,
    )


if __name__ == "__main__":
    main()

