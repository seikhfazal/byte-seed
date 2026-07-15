from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 1


class CheckpointKind(str, Enum):
    PRETRAIN = "pretrain"
    SFT = "sft"
    MODEL_ONLY = "model_only"


class CheckpointOperation(str, Enum):
    PRETRAIN_RESUME = "pretraining resume"
    MODEL_LOAD = "model loading"


class CheckpointError(RuntimeError):
    """Base error for checkpoint container and compatibility failures."""


class CheckpointLoadError(CheckpointError):
    """Raised when a checkpoint file cannot be decoded."""


class CheckpointValidationError(CheckpointError):
    """Raised when a checkpoint container is malformed or unsupported."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a valid checkpoint cannot serve the requested operation."""


@dataclass(frozen=True)
class CheckpointInfo:
    version: int | None
    kind: CheckpointKind | None
    legacy: bool
    progress: int | None

    @property
    def kind_label(self) -> str:
        if self.kind is not None:
            suffix = " (legacy)" if self.legacy else ""
            return f"{self.kind.value}{suffix}"
        return "ambiguous legacy" if self.legacy else "unknown"


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    data: dict[str, Any]
    info: CheckpointInfo


def build_checkpoint(
    kind: CheckpointKind | str,
    *,
    model_state: Mapping[str, Any],
    config: Mapping[str, Any],
    iteration: int | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    best_val: float | None = None,
) -> dict[str, Any]:
    """Build a version-1 checkpoint while retaining ByteSeed's legacy payload keys."""
    checkpoint_kind = _coerce_kind(kind)
    data: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_kind": checkpoint_kind.value,
        "model": model_state,
        "config": dict(config),
    }
    if iteration is not None:
        data["iter"] = _validate_iteration(iteration)
    if optimizer_state is not None:
        data["optimizer"] = optimizer_state
    if best_val is not None:
        data["best_val"] = float(best_val)

    required = {
        CheckpointKind.PRETRAIN: ("optimizer", "iter"),
        CheckpointKind.SFT: ("iter",),
        CheckpointKind.MODEL_ONLY: (),
    }[checkpoint_kind]
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(
            f"Cannot build {checkpoint_kind.value} checkpoint; missing required fields: "
            f"{', '.join(missing)}."
        )
    return data


def classify_checkpoint(data: Mapping[str, Any]) -> CheckpointInfo:
    """Classify explicit metadata first, then known legacy structures conservatively."""
    has_version = "checkpoint_version" in data
    has_kind = "checkpoint_kind" in data
    if has_version != has_kind:
        raise CheckpointValidationError(
            "Checkpoint metadata is incomplete; checkpoint_version and checkpoint_kind must appear together."
        )

    if has_version:
        version = data["checkpoint_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise CheckpointValidationError("checkpoint_version must be an integer.")
        if version != CHECKPOINT_VERSION:
            raise CheckpointValidationError(
                f"Unsupported checkpoint schema version {version}; this ByteSeed build supports "
                f"version {CHECKPOINT_VERSION}."
            )
        kind = _coerce_kind(data["checkpoint_kind"], error_type=CheckpointValidationError)
        _validate_model_state(data)
        if not isinstance(data.get("config"), Mapping):
            raise CheckpointValidationError(
                f"Version {version} checkpoint kind '{kind.value}' requires a mapping field 'config'."
            )
        return CheckpointInfo(
            version=version,
            kind=kind,
            legacy=False,
            progress=_progress(data),
        )

    _validate_model_state(data)
    if _has_valid_pretrain_resume_fields(data):
        kind = CheckpointKind.PRETRAIN
    elif "optimizer" in data:
        # An optimizer without all current resume fields is ambiguous and must fail closed for resume.
        kind = None
    else:
        # Known Anchor/SFT and bare state-dict containers are model-bearing but not resumable.
        kind = CheckpointKind.MODEL_ONLY
    return CheckpointInfo(version=None, kind=kind, legacy=True, progress=_progress(data))


def load_checkpoint(
    path: str | Path,
    operation: CheckpointOperation,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Load and validate an explicit checkpoint without any fallback behavior."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found for requested operation '{operation.value}': {checkpoint_path}"
        )
    data = _read_checkpoint(checkpoint_path, map_location=map_location)
    info = classify_checkpoint(data)
    _require_compatible(data, info, operation, checkpoint_path)
    return LoadedCheckpoint(path=checkpoint_path, data=data, info=info)


def discover_checkpoint(
    checkpoint_dir: str | Path,
    operation: CheckpointOperation,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint | None:
    """Find the best compatible checkpoint, ignoring corrupt or incompatible candidates."""
    directory = Path(checkpoint_dir)
    candidates: list[LoadedCheckpoint] = []
    for path in sorted(directory.glob("*.pt"), key=_normalized_path):
        try:
            candidates.append(load_checkpoint(path, operation, map_location=map_location))
        except (CheckpointError, FileNotFoundError):
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: _selection_key(candidate, operation))


def select_checkpoint(
    checkpoint_dir: str | Path,
    operation: CheckpointOperation,
    *,
    explicit_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint | None:
    """Honor an explicit path or deterministically discover a compatible checkpoint."""
    if explicit_path is not None:
        return load_checkpoint(explicit_path, operation, map_location=map_location)
    return discover_checkpoint(checkpoint_dir, operation, map_location=map_location)


def _read_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    try:
        raw = torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
        raise CheckpointLoadError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise CheckpointValidationError(
            f"Checkpoint {path} must contain a mapping, got {type(raw).__name__}."
        )
    return dict(raw)


def _require_compatible(
    data: Mapping[str, Any],
    info: CheckpointInfo,
    operation: CheckpointOperation,
    path: Path,
) -> None:
    if operation is CheckpointOperation.MODEL_LOAD:
        return

    missing = [field for field in ("model", "optimizer", "config", "iter") if field not in data]
    invalid: list[str] = []
    if "optimizer" in data and not isinstance(data["optimizer"], Mapping):
        invalid.append("optimizer must be a mapping")
    if "config" in data and not isinstance(data["config"], Mapping):
        invalid.append("config must be a mapping")
    if "iter" in data and not _is_valid_iteration(data["iter"]):
        invalid.append("iter must be a non-negative integer")

    wrong_kind = info.kind is not CheckpointKind.PRETRAIN
    if wrong_kind or missing or invalid:
        details = [f"detected checkpoint kind: {info.kind_label}"]
        if missing:
            details.append(f"missing required fields: {', '.join(missing)}")
        if invalid:
            details.append(f"invalid fields: {', '.join(invalid)}")
        raise CheckpointCompatibilityError(
            f"Checkpoint {path} is incompatible with requested operation '{operation.value}'; "
            + "; ".join(details)
            + "."
        )


def _validate_model_state(data: Mapping[str, Any]) -> None:
    if "model" not in data:
        raise CheckpointValidationError("Checkpoint is malformed: missing required field 'model'.")
    if not isinstance(data["model"], Mapping):
        raise CheckpointValidationError("Checkpoint is malformed: field 'model' must be a mapping.")


def _has_valid_pretrain_resume_fields(data: Mapping[str, Any]) -> bool:
    return (
        isinstance(data.get("model"), Mapping)
        and isinstance(data.get("optimizer"), Mapping)
        and isinstance(data.get("config"), Mapping)
        and _is_valid_iteration(data.get("iter"))
    )


def _progress(data: Mapping[str, Any]) -> int | None:
    value = data.get("iter")
    return int(value) if _is_valid_iteration(value) else None


def _validate_iteration(value: int) -> int:
    if not _is_valid_iteration(value):
        raise ValueError(f"iteration must be a non-negative integer, got {value!r}.")
    return int(value)


def _is_valid_iteration(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _coerce_kind(
    value: CheckpointKind | str,
    *,
    error_type: type[Exception] = ValueError,
) -> CheckpointKind:
    try:
        return value if isinstance(value, CheckpointKind) else CheckpointKind(str(value))
    except ValueError as exc:
        allowed = ", ".join(kind.value for kind in CheckpointKind)
        raise error_type(f"Unsupported checkpoint kind {value!r}; expected one of: {allowed}.") from exc


def _normalized_path(path: Path) -> str:
    return path.as_posix().casefold()


def _selection_key(
    checkpoint: LoadedCheckpoint,
    operation: CheckpointOperation,
) -> tuple[int, str]:
    normalized_path = _normalized_path(checkpoint.path)
    if operation is CheckpointOperation.PRETRAIN_RESUME:
        # Pretraining progress is comparable; path ordering is the stable equal-progress tie-breaker.
        return checkpoint.info.progress or 0, normalized_path
    # Iteration counters are not comparable across pretraining and stage-local SFT.
    return checkpoint.path.stat().st_mtime_ns, normalized_path
