from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from byteseed.checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointCompatibilityError,
    CheckpointKind,
    CheckpointOperation,
    CheckpointValidationError,
    build_checkpoint,
    discover_checkpoint,
    load_checkpoint,
    select_checkpoint,
)


def _model_state() -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([1.0])}


def _optimizer_state() -> dict[str, object]:
    return {"state": {}, "param_groups": []}


def _pretrain(iteration: int) -> dict[str, object]:
    return build_checkpoint(
        CheckpointKind.PRETRAIN,
        model_state=_model_state(),
        optimizer_state=_optimizer_state(),
        config={"model_name": "tiny"},
        iteration=iteration,
    )


def _sft(iteration: int) -> dict[str, object]:
    return build_checkpoint(
        CheckpointKind.SFT,
        model_state=_model_state(),
        config={"model_name": "tiny"},
        iteration=iteration,
    )


def _model_only() -> dict[str, object]:
    return build_checkpoint(
        CheckpointKind.MODEL_ONLY,
        model_state=_model_state(),
        config={"model_name": "tiny"},
    )


def _save(path: Path, checkpoint: object, *, mtime: int | None = None) -> Path:
    torch.save(checkpoint, path)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_pretrain_discovery_ignores_newer_sft_checkpoint(tmp_path):
    pretrain_path = _save(tmp_path / "pretrain.pt", _pretrain(8), mtime=100)
    _save(tmp_path / "newer-sft.pt", _sft(200), mtime=200)

    selected = discover_checkpoint(tmp_path, CheckpointOperation.PRETRAIN_RESUME)

    assert selected is not None
    assert selected.path == pretrain_path


def test_pretrain_discovery_ignores_newer_model_only_checkpoint(tmp_path):
    pretrain_path = _save(tmp_path / "pretrain.pt", _pretrain(8), mtime=100)
    _save(tmp_path / "newer-model.pt", _model_only(), mtime=200)

    selected = discover_checkpoint(tmp_path, CheckpointOperation.PRETRAIN_RESUME)

    assert selected is not None
    assert selected.path == pretrain_path


def test_marked_pretrain_without_optimizer_is_not_resumable(tmp_path):
    checkpoint_path = _save(
        tmp_path / "incomplete.pt",
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_kind": "pretrain",
            "model": _model_state(),
            "config": {"model_name": "tiny"},
            "iter": 9,
        },
    )

    assert discover_checkpoint(tmp_path, CheckpointOperation.PRETRAIN_RESUME) is None
    with pytest.raises(CheckpointCompatibilityError, match="missing required fields: optimizer"):
        load_checkpoint(checkpoint_path, CheckpointOperation.PRETRAIN_RESUME)


def test_pretrain_discovery_selects_highest_progress(tmp_path):
    _save(tmp_path / "newer-file.pt", _pretrain(4), mtime=300)
    highest_path = _save(tmp_path / "older-file.pt", _pretrain(12), mtime=100)

    selected = discover_checkpoint(tmp_path, CheckpointOperation.PRETRAIN_RESUME)

    assert selected is not None
    assert selected.path == highest_path
    assert selected.info.progress == 12


def test_equal_progress_uses_normalized_path_tie_breaker_not_creation_order(tmp_path):
    expected = _save(tmp_path / "z-last.pt", _pretrain(6))
    _save(tmp_path / "a-first.pt", _pretrain(6))

    selected = discover_checkpoint(tmp_path, CheckpointOperation.PRETRAIN_RESUME)

    assert selected is not None
    assert selected.path == expected


def test_no_compatible_pretrain_checkpoint_returns_none(tmp_path):
    _save(tmp_path / "sft.pt", _sft(3))
    _save(tmp_path / "model.pt", _model_only())

    assert discover_checkpoint(tmp_path, CheckpointOperation.PRETRAIN_RESUME) is None


def test_explicit_valid_pretrain_checkpoint_is_accepted(tmp_path):
    checkpoint_path = _save(tmp_path / "resume.pt", _pretrain(11))

    selected = select_checkpoint(
        tmp_path,
        CheckpointOperation.PRETRAIN_RESUME,
        explicit_path=checkpoint_path,
    )

    assert selected is not None
    assert selected.path == checkpoint_path
    assert selected.info.kind is CheckpointKind.PRETRAIN


def test_explicit_sft_checkpoint_for_pretrain_resume_fails_clearly(tmp_path):
    checkpoint_path = _save(tmp_path / "sft.pt", _sft(4))

    with pytest.raises(CheckpointCompatibilityError) as exc_info:
        select_checkpoint(
            tmp_path,
            CheckpointOperation.PRETRAIN_RESUME,
            explicit_path=checkpoint_path,
        )

    message = str(exc_info.value)
    assert "pretraining resume" in message
    assert "detected checkpoint kind: sft" in message
    assert "missing required fields: optimizer" in message


def test_explicit_malformed_checkpoint_fails_validation(tmp_path):
    checkpoint_path = _save(tmp_path / "malformed.pt", {"config": {}})

    with pytest.raises(CheckpointValidationError, match="missing required field 'model'"):
        load_checkpoint(checkpoint_path, CheckpointOperation.MODEL_LOAD)


def test_explicit_missing_path_does_not_fall_back(tmp_path):
    _save(tmp_path / "valid.pt", _pretrain(5))
    missing = tmp_path / "requested-but-missing.pt"

    with pytest.raises(FileNotFoundError, match="pretraining resume"):
        select_checkpoint(
            tmp_path,
            CheckpointOperation.PRETRAIN_RESUME,
            explicit_path=missing,
        )


def test_legacy_anchor_like_checkpoint_remains_valid_for_model_loading(tmp_path):
    checkpoint_path = _save(
        tmp_path / "anchor.pt",
        {"model": _model_state(), "config": {"model_name": "tiny"}, "iter": 80},
    )

    loaded = load_checkpoint(checkpoint_path, CheckpointOperation.MODEL_LOAD)

    assert loaded.info.legacy is True
    assert loaded.info.kind is CheckpointKind.MODEL_ONLY


def test_structurally_complete_legacy_pretrain_checkpoint_is_resumable(tmp_path):
    checkpoint_path = _save(
        tmp_path / "legacy-pretrain.pt",
        {
            "model": _model_state(),
            "optimizer": _optimizer_state(),
            "config": {"model_name": "tiny"},
            "iter": 5,
        },
    )

    loaded = load_checkpoint(checkpoint_path, CheckpointOperation.PRETRAIN_RESUME)

    assert loaded.info.legacy is True
    assert loaded.info.kind is CheckpointKind.PRETRAIN


def test_legacy_model_only_checkpoint_is_not_pretrain_resumable(tmp_path):
    checkpoint_path = _save(tmp_path / "legacy-model.pt", {"model": _model_state()})

    with pytest.raises(CheckpointCompatibilityError) as exc_info:
        load_checkpoint(checkpoint_path, CheckpointOperation.PRETRAIN_RESUME)

    assert "model_only (legacy)" in str(exc_info.value)
    assert "optimizer, config, iter" in str(exc_info.value)


def test_ambiguous_legacy_optimizer_checkpoint_fails_closed_for_resume(tmp_path):
    checkpoint_path = _save(
        tmp_path / "ambiguous.pt",
        {
            "model": _model_state(),
            "optimizer": _optimizer_state(),
            "config": {"model_name": "tiny"},
        },
    )

    with pytest.raises(CheckpointCompatibilityError) as exc_info:
        load_checkpoint(checkpoint_path, CheckpointOperation.PRETRAIN_RESUME)

    assert "ambiguous legacy" in str(exc_info.value)
    assert "missing required fields: iter" in str(exc_info.value)


def test_corrupt_candidate_does_not_hide_valid_checkpoint(tmp_path):
    (tmp_path / "corrupt.pt").write_bytes(b"not a torch checkpoint")
    valid_path = _save(tmp_path / "valid.pt", _pretrain(2))

    selected = discover_checkpoint(tmp_path, CheckpointOperation.PRETRAIN_RESUME)

    assert selected is not None
    assert selected.path == valid_path


def test_unrelated_file_types_are_ignored(tmp_path):
    _save(tmp_path / "checkpoint.txt", _pretrain(2))

    assert discover_checkpoint(tmp_path, CheckpointOperation.PRETRAIN_RESUME) is None
