from __future__ import annotations

import os

import torch

from byteseed.checkpoint import (
    CheckpointKind,
    CheckpointOperation,
    build_checkpoint,
    discover_checkpoint,
)


def _checkpoint(kind: CheckpointKind, iteration: int | None = None) -> dict[str, object]:
    return build_checkpoint(
        kind,
        model_state={"weight": torch.tensor([1.0])},
        config={"model_name": "tiny"},
        iteration=iteration,
    )


def test_model_load_discovery_preserves_newest_mtime_behavior(tmp_path):
    older = tmp_path / "older.pt"
    newer = tmp_path / "newer.pt"
    torch.save(_checkpoint(CheckpointKind.MODEL_ONLY), older)
    torch.save(_checkpoint(CheckpointKind.SFT, iteration=1), newer)
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))

    selected = discover_checkpoint(tmp_path, CheckpointOperation.MODEL_LOAD)

    assert selected is not None
    assert selected.path == newer


def test_model_load_equal_mtime_uses_normalized_path_tie_breaker(tmp_path):
    expected = tmp_path / "z-last.pt"
    other = tmp_path / "a-first.pt"
    torch.save(_checkpoint(CheckpointKind.MODEL_ONLY), expected)
    torch.save(_checkpoint(CheckpointKind.MODEL_ONLY), other)
    os.utime(expected, (100, 100))
    os.utime(other, (100, 100))

    selected = discover_checkpoint(tmp_path, CheckpointOperation.MODEL_LOAD)

    assert selected is not None
    assert selected.path == expected
