from __future__ import annotations

import pytest
import torch

from byteseed.checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointKind,
    CheckpointOperation,
    CheckpointValidationError,
    build_checkpoint,
    load_checkpoint,
)


def _model_state() -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([1.0])}


def _optimizer_state() -> dict[str, object]:
    return {"state": {}, "param_groups": []}


def test_pretrain_checkpoint_metadata_contains_required_resume_state():
    checkpoint = build_checkpoint(
        CheckpointKind.PRETRAIN,
        model_state=_model_state(),
        optimizer_state=_optimizer_state(),
        config={"model_name": "tiny"},
        iteration=12,
        best_val=0.25,
    )

    assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
    assert checkpoint["checkpoint_kind"] == "pretrain"
    assert checkpoint["model"]["weight"].item() == 1.0
    assert checkpoint["optimizer"] == _optimizer_state()
    assert checkpoint["config"] == {"model_name": "tiny"}
    assert checkpoint["iter"] == 12
    assert checkpoint["best_val"] == 0.25


def test_sft_checkpoint_metadata_identifies_sft_kind():
    checkpoint = build_checkpoint(
        CheckpointKind.SFT,
        model_state=_model_state(),
        config={"model_name": "tiny"},
        iteration=7,
    )

    assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
    assert checkpoint["checkpoint_kind"] == "sft"
    assert checkpoint["iter"] == 7
    assert "optimizer" not in checkpoint


def test_model_only_checkpoint_metadata_identifies_model_only_kind():
    checkpoint = build_checkpoint(
        CheckpointKind.MODEL_ONLY,
        model_state=_model_state(),
        config={"model_name": "tiny"},
    )

    assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
    assert checkpoint["checkpoint_kind"] == "model_only"
    assert "iter" not in checkpoint
    assert "optimizer" not in checkpoint


def test_pretrain_checkpoint_builder_rejects_missing_optimizer():
    with pytest.raises(ValueError, match="missing required fields: optimizer"):
        build_checkpoint(
            CheckpointKind.PRETRAIN,
            model_state=_model_state(),
            config={"model_name": "tiny"},
            iteration=3,
        )


def test_explicit_future_schema_version_fails_clearly(tmp_path):
    checkpoint_path = tmp_path / "future.pt"
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION + 1,
            "checkpoint_kind": "model_only",
            "model": _model_state(),
            "config": {"model_name": "tiny"},
        },
        checkpoint_path,
    )

    with pytest.raises(CheckpointValidationError, match="Unsupported checkpoint schema version"):
        load_checkpoint(checkpoint_path, CheckpointOperation.MODEL_LOAD)


def test_partial_version_metadata_fails_clearly(tmp_path):
    checkpoint_path = tmp_path / "partial.pt"
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model": _model_state(),
            "config": {"model_name": "tiny"},
        },
        checkpoint_path,
    )

    with pytest.raises(CheckpointValidationError, match="checkpoint_version and checkpoint_kind"):
        load_checkpoint(checkpoint_path, CheckpointOperation.MODEL_LOAD)
