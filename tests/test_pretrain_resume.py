from __future__ import annotations

from pathlib import Path

import pytest
import torch

from byteseed.checkpoint import (
    RESUME_STATE_VERSION,
    CheckpointCompatibilityError,
    CheckpointKind,
    CheckpointValidationError,
    build_checkpoint,
    build_resume_state,
    training_config_snapshot,
)
from byteseed.pretrain import (
    next_iteration,
    resolve_resume_checkpoint,
    should_evaluate,
    update_early_stopping,
)


def _model_state() -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([1.0])}


def _optimizer_state() -> dict[str, object]:
    return {"state": {}, "param_groups": []}


def _partial_checkpoint(tiny_config, iteration: int) -> dict[str, object]:
    return build_checkpoint(
        CheckpointKind.PRETRAIN,
        model_state=_model_state(),
        optimizer_state=_optimizer_state(),
        config=tiny_config.__dict__,
        iteration=iteration,
        best_val=0.5,
    )


def _exact_checkpoint(
    tiny_config, iteration: int, checkpoint_provenance
) -> dict[str, object]:
    critical = training_config_snapshot(
        tiny_config.__dict__,
        device_type="cpu",
        amp_enabled=False,
    )
    resume_state = build_resume_state(
        scaler=None,
        best_val=0.5,
        patience_left=2,
        training_config=critical,
    )
    return build_checkpoint(
        CheckpointKind.PRETRAIN,
        model_state=_model_state(),
        optimizer_state=_optimizer_state(),
        config=tiny_config.__dict__,
        iteration=iteration,
        best_val=0.5,
        resume_state=resume_state,
        provenance=checkpoint_provenance,
    )


def _save(path: Path, payload: object) -> Path:
    torch.save(payload, path)
    return path


def _resolve(
    checkpoint_dir,
    *,
    checkpoint_provenance,
    explicit_path,
    allow_inexact_resume,
):
    return resolve_resume_checkpoint(
        checkpoint_dir,
        explicit_path=explicit_path,
        allow_inexact_resume=allow_inexact_resume,
        runtime_tokenizer_identity=checkpoint_provenance["tokenizer"],
        runtime_data_manifest=checkpoint_provenance["data_manifest"],
    )


def test_auto_resume_selects_exact_checkpoint_over_higher_progress_partial(
    tmp_path,
    tiny_config,
    checkpoint_provenance,
):
    exact_path = _save(
        tmp_path / "exact.pt",
        _exact_checkpoint(tiny_config, 4, checkpoint_provenance),
    )
    _save(tmp_path / "partial.pt", _partial_checkpoint(tiny_config, 99))

    selected, resume_state = _resolve(
        tmp_path,
        checkpoint_provenance=checkpoint_provenance,
        explicit_path=None,
        allow_inexact_resume=False,
    )

    assert selected is not None
    assert selected.path == exact_path
    assert resume_state is not None
    assert resume_state["version"] == RESUME_STATE_VERSION


def test_invalid_exact_candidate_does_not_hide_valid_exact_candidate(
    tmp_path, tiny_config, checkpoint_provenance
):
    valid_path = _save(
        tmp_path / "valid.pt",
        _exact_checkpoint(tiny_config, 4, checkpoint_provenance),
    )
    invalid = _exact_checkpoint(tiny_config, 99, checkpoint_provenance)
    invalid["resume_state"]["version"] = RESUME_STATE_VERSION + 1
    _save(tmp_path / "invalid.pt", invalid)

    selected, resume_state = _resolve(
        tmp_path,
        checkpoint_provenance=checkpoint_provenance,
        explicit_path=None,
        allow_inexact_resume=False,
    )

    assert selected is not None
    assert selected.path == valid_path
    assert resume_state is not None

def test_explicit_exact_checkpoint_is_accepted(
    tmp_path, tiny_config, checkpoint_provenance
):
    exact_path = _save(
        tmp_path / "exact.pt",
        _exact_checkpoint(tiny_config, 4, checkpoint_provenance),
    )

    selected, resume_state = _resolve(
        tmp_path,
        checkpoint_provenance=checkpoint_provenance,
        explicit_path=exact_path,
        allow_inexact_resume=False,
    )

    assert selected is not None
    assert selected.path == exact_path
    assert resume_state is not None


def test_exact_checkpoint_without_backend_metadata_defaults_to_manual(
    tmp_path, tiny_config, checkpoint_provenance
):
    checkpoint = _exact_checkpoint(tiny_config, 4, checkpoint_provenance)
    checkpoint["config"].pop("attention_backend")
    checkpoint["resume_state"]["training_config"].pop("attention_backend")
    checkpoint_path = _save(tmp_path / "pre-sdpa-exact.pt", checkpoint)

    selected, resume_state = _resolve(
        tmp_path,
        checkpoint_provenance=checkpoint_provenance,
        explicit_path=checkpoint_path,
        allow_inexact_resume=False,
    )

    assert selected is not None
    assert resume_state is not None


def test_partial_explicit_resume_without_opt_in_fails(
    tmp_path, tiny_config, checkpoint_provenance
):
    partial_path = _save(tmp_path / "partial.pt", _partial_checkpoint(tiny_config, 4))

    with pytest.raises(CheckpointCompatibilityError, match="only an inexact.*allow-inexact"):
        _resolve(
            tmp_path,
            checkpoint_provenance=checkpoint_provenance,
            explicit_path=partial_path,
            allow_inexact_resume=False,
        )


def test_partial_explicit_resume_with_opt_in_warns(
    tmp_path, tiny_config, capsys, checkpoint_provenance
):
    partial_path = _save(tmp_path / "partial.pt", _partial_checkpoint(tiny_config, 4))

    selected, resume_state = _resolve(
        tmp_path,
        checkpoint_provenance=checkpoint_provenance,
        explicit_path=partial_path,
        allow_inexact_resume=True,
    )

    assert selected is not None
    assert selected.path == partial_path
    assert resume_state is None
    warning = capsys.readouterr().out
    assert "WARNING: inexact pretraining resume explicitly enabled" in warning
    assert "not exact because" in warning


def test_auto_resume_does_not_silently_downgrade_to_partial(
    tmp_path, tiny_config, checkpoint_provenance
):
    partial_path = _save(tmp_path / "partial.pt", _partial_checkpoint(tiny_config, 4))

    with pytest.raises(CheckpointCompatibilityError, match="never downgrades") as exc_info:
        _resolve(
            tmp_path,
            checkpoint_provenance=checkpoint_provenance,
            explicit_path=None,
            allow_inexact_resume=False,
        )

    assert str(partial_path) in str(exc_info.value)


def test_inexact_opt_in_requires_explicit_path(tmp_path, checkpoint_provenance):
    with pytest.raises(ValueError, match="requires --resume-checkpoint"):
        _resolve(
            tmp_path,
            checkpoint_provenance=checkpoint_provenance,
            explicit_path=None,
            allow_inexact_resume=True,
        )


def test_malformed_claimed_exact_state_is_not_downgraded_by_opt_in(
    tmp_path, tiny_config, checkpoint_provenance
):
    checkpoint = _exact_checkpoint(tiny_config, 4, checkpoint_provenance)
    checkpoint["resume_state"]["version"] = RESUME_STATE_VERSION + 1
    checkpoint_path = _save(tmp_path / "future-resume.pt", checkpoint)

    with pytest.raises(CheckpointValidationError, match="Unsupported resume-state version"):
        _resolve(
            tmp_path,
            checkpoint_provenance=checkpoint_provenance,
            explicit_path=checkpoint_path,
            allow_inexact_resume=True,
        )


def test_empty_auto_resume_directory_returns_no_checkpoint(
    tmp_path, checkpoint_provenance
):
    selected, resume_state = _resolve(
        tmp_path,
        checkpoint_provenance=checkpoint_provenance,
        explicit_path=None,
        allow_inexact_resume=False,
    )

    assert selected is None
    assert resume_state is None


@pytest.mark.parametrize(
    ("completed", "expected"),
    [(0, 1), (7, 8), (25, 26)],
)
def test_completed_iteration_resumes_at_next_step(completed, expected):
    assert next_iteration(completed) == expected


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_completed_iteration_rejects_invalid_values(invalid):
    with pytest.raises(ValueError):
        next_iteration(invalid)


def test_resumed_step_range_neither_repeats_nor_skips():
    assert list(range(next_iteration(4), 8)) == [5, 6, 7]

def test_evaluation_cadence_stays_aligned_after_resume(tiny_config):
    tiny_config.eval_interval = 3
    tiny_config.max_iters = 10

    evaluated = [step for step in range(next_iteration(4), tiny_config.max_iters) if should_evaluate(step, tiny_config)]

    assert evaluated == [6, 9]


def test_early_stopping_patience_continues_without_reset():
    best_val = 0.5
    patience_left = 2

    best_val, patience_left, improved, stopped = update_early_stopping(
        0.6,
        best_val,
        patience_left,
        configured_patience=2,
    )
    assert (best_val, patience_left, improved, stopped) == (0.5, 1, False, False)

    resumed_result = update_early_stopping(
        0.7,
        best_val,
        patience_left,
        configured_patience=2,
    )

    assert resumed_result == (0.5, 0, False, True)


def test_early_stopping_improvement_restores_full_patience():
    result = update_early_stopping(
        validation_loss=0.4,
        best_val=0.5,
        patience_left=1,
        configured_patience=3,
    )

    assert result == (0.4, 3, True, False)


def test_disabled_early_stopping_does_not_decrement():
    result = update_early_stopping(
        validation_loss=0.6,
        best_val=0.5,
        patience_left=0,
        configured_patience=0,
    )

    assert result == (0.5, 0, False, False)
