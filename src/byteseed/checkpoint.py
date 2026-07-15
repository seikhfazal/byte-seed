from __future__ import annotations

import pickle
import random
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 1
RESUME_STATE_VERSION = 1

TRAINING_CRITICAL_CONFIG_FIELDS = (
    "vocab_size",
    "block_size",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "max_iters",
    "eval_interval",
    "eval_iters",
    "weight_decay",
    "warmup_iters",
    "processed_data_dir",
    "train_split",
    "seed",
    "early_stopping_patience",
)

_TRAINING_RUNTIME_DEFAULTS = {
    "optimizer": "AdamW",
    "optimizer_betas": (0.9, 0.999),
    "optimizer_eps": 1e-8,
    "learning_rate_schedule": "linear_warmup_then_constant",
    "gradient_clip_norm": 1.0,
    "batch_sampler": "global_torch_randint",
    "autocast_dtype": "float16",
}


class CheckpointKind(str, Enum):
    PRETRAIN = "pretrain"
    SFT = "sft"
    MODEL_ONLY = "model_only"


class CheckpointOperation(str, Enum):
    PRETRAIN_RESUME = "pretraining resume"
    PRETRAIN_EXACT_RESUME = "exact pretraining resume"
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
    resume_state: Mapping[str, Any] | None = None,
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
    if resume_state is not None:
        if checkpoint_kind is not CheckpointKind.PRETRAIN:
            raise ValueError("Only pretraining checkpoints may contain exact resume state.")
        data["resume_state"] = dict(resume_state)

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
    if resume_state is not None:
        validate_exact_resume_checkpoint(data)
    return data


def capture_rng_state() -> dict[str, Any]:
    """Capture only RNG sources used by pretraining without initializing CUDA."""
    cuda_states = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore validated RNG state at the final point before stochastic work."""
    python_state, cpu_state, cuda_states = _validated_rng_state(state)
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError(
                "Exact resume requires saved CUDA RNG states, but CUDA is unavailable."
            )
        device_count = torch.cuda.device_count()
        if len(cuda_states) != device_count:
            raise CheckpointCompatibilityError(
                "Exact resume CUDA device-count mismatch: "
                f"checkpoint={len(cuda_states)}, current={device_count}."
            )

    random.setstate(python_state)
    torch.set_rng_state(cpu_state.cpu())
    if cuda_states is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def capture_scaler_state(scaler: Any | None) -> dict[str, Any]:
    """Represent both active CUDA scaling and disabled CPU/no-scaler operation."""
    enabled = bool(scaler is not None and scaler.is_enabled())
    return {
        "enabled": enabled,
        "state": scaler.state_dict() if enabled else {},
    }


def restore_scaler_state(scaler: Any | None, state: Mapping[str, Any]) -> None:
    """Restore an active scaler, rejecting enabled/disabled configuration drift."""
    expected_enabled, saved_state = _validated_scaler_state(state)
    actual_enabled = bool(scaler is not None and scaler.is_enabled())
    if actual_enabled != expected_enabled:
        raise CheckpointCompatibilityError(
            "Exact resume AMP configuration mismatch: "
            f"checkpoint enabled={expected_enabled}, current enabled={actual_enabled}."
        )
    if expected_enabled:
        scaler.load_state_dict(saved_state)


def training_config_snapshot(
    config: Mapping[str, Any],
    *,
    device_type: str,
    amp_enabled: bool,
) -> dict[str, Any]:
    """Extract configuration that can change pretraining continuation."""
    missing = [field for field in TRAINING_CRITICAL_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(
            "Cannot capture exact-resume training configuration; missing fields: "
            + ", ".join(missing)
            + "."
        )
    snapshot = {field: config[field] for field in TRAINING_CRITICAL_CONFIG_FIELDS}
    snapshot.update(_TRAINING_RUNTIME_DEFAULTS)
    snapshot["device_type"] = str(device_type)
    snapshot["amp_enabled"] = bool(amp_enabled)
    return snapshot


def validate_training_config(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    """Reject exact resume when a training-critical value differs."""
    if not isinstance(saved, Mapping):
        raise CheckpointValidationError("resume_state.training_config must be a mapping.")
    if not isinstance(current, Mapping):
        raise TypeError("current training configuration must be a mapping.")

    required = set(TRAINING_CRITICAL_CONFIG_FIELDS) | set(_TRAINING_RUNTIME_DEFAULTS) | {
        "device_type",
        "amp_enabled",
    }
    missing_saved = sorted(required - set(saved))
    missing_current = sorted(required - set(current))
    if missing_saved or missing_current:
        details = []
        if missing_saved:
            details.append("checkpoint missing " + ", ".join(missing_saved))
        if missing_current:
            details.append("current run missing " + ", ".join(missing_current))
        raise CheckpointCompatibilityError(
            "Exact resume training configuration is incomplete: " + "; ".join(details) + "."
        )

    differences = [field for field in sorted(required) if saved[field] != current[field]]
    if differences:
        details = ", ".join(
            f"{field} (checkpoint={_safe_value(saved[field])}, current={_safe_value(current[field])})"
            for field in differences
        )
        raise CheckpointCompatibilityError(
            "Exact resume training configuration mismatch: " + details + "."
        )


def build_resume_state(
    *,
    scaler: Any | None,
    best_val: float,
    patience_left: int,
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture a coherent version-1 exact pretraining continuation point."""
    if isinstance(patience_left, bool) or not isinstance(patience_left, int) or patience_left < 0:
        raise ValueError("patience_left must be a non-negative integer.")
    return {
        "version": RESUME_STATE_VERSION,
        "rng_state": capture_rng_state(),
        "amp_scaler": capture_scaler_state(scaler),
        "early_stopping": {
            "best_val": float(best_val),
            "patience_left": patience_left,
        },
        "training_config": dict(training_config),
    }


def validate_exact_resume_checkpoint(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and return the exact-resume block of a pretraining checkpoint."""
    info = classify_checkpoint(data)
    if info.kind is not CheckpointKind.PRETRAIN:
        raise CheckpointCompatibilityError(
            f"Exact pretraining resume requires kind 'pretrain'; detected {info.kind_label}."
        )
    structural_missing = [
        field for field in ("model", "optimizer", "config", "iter") if field not in data
    ]
    structural_invalid = []
    if "optimizer" in data and not isinstance(data["optimizer"], Mapping):
        structural_invalid.append("optimizer must be a mapping")
    if "iter" in data and not _is_valid_iteration(data["iter"]):
        structural_invalid.append("iter must be a non-negative integer")
    if structural_missing or structural_invalid:
        details = []
        if structural_missing:
            details.append("missing " + ", ".join(structural_missing))
        if structural_invalid:
            details.extend(structural_invalid)
        raise CheckpointCompatibilityError(
            "Exact pretraining resume requires complete structural state: "
            + "; ".join(details)
            + "."
        )
    if "resume_state" not in data:
        raise CheckpointCompatibilityError(
            "Checkpoint is structurally pretraining-resumable but lacks exact resume_state."
        )
    resume_state = data["resume_state"]
    if not isinstance(resume_state, Mapping):
        raise CheckpointValidationError("resume_state must be a mapping.")

    version = resume_state.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise CheckpointValidationError("resume_state.version must be an integer.")
    if version != RESUME_STATE_VERSION:
        raise CheckpointValidationError(
            f"Unsupported resume-state version {version}; this ByteSeed build supports "
            f"version {RESUME_STATE_VERSION}."
        )

    required = {"rng_state", "amp_scaler", "early_stopping", "training_config"}
    missing = sorted(required - set(resume_state))
    if missing:
        raise CheckpointValidationError(
            "resume_state is incomplete; missing required fields: " + ", ".join(missing) + "."
        )
    _validated_rng_state(resume_state["rng_state"])
    _validated_scaler_state(resume_state["amp_scaler"])

    early_stopping = resume_state["early_stopping"]
    if not isinstance(early_stopping, Mapping):
        raise CheckpointValidationError("resume_state.early_stopping must be a mapping.")
    missing_early = sorted({"best_val", "patience_left"} - set(early_stopping))
    if missing_early:
        raise CheckpointValidationError(
            "resume_state.early_stopping is incomplete; missing: "
            + ", ".join(missing_early)
            + "."
        )
    best_val = early_stopping["best_val"]
    if isinstance(best_val, bool) or not isinstance(best_val, (int, float)):
        raise CheckpointValidationError(
            "resume_state.early_stopping.best_val must be numeric."
        )
    patience_left = early_stopping["patience_left"]
    if isinstance(patience_left, bool) or not isinstance(patience_left, int) or patience_left < 0:
        raise CheckpointValidationError(
            "resume_state.early_stopping.patience_left must be a non-negative integer."
        )
    training_config = resume_state["training_config"]
    if not isinstance(training_config, Mapping):
        raise CheckpointValidationError("resume_state.training_config must be a mapping.")
    required_training = (
        set(TRAINING_CRITICAL_CONFIG_FIELDS)
        | set(_TRAINING_RUNTIME_DEFAULTS)
        | {"device_type", "amp_enabled"}
    )
    missing_training = sorted(required_training - set(training_config))
    if missing_training:
        raise CheckpointValidationError(
            "resume_state.training_config is incomplete; missing: "
            + ", ".join(missing_training)
            + "."
        )

    top_level_best = data.get("best_val")
    if top_level_best is not None:
        if isinstance(top_level_best, bool) or not isinstance(top_level_best, (int, float)):
            raise CheckpointValidationError("Checkpoint best_val must be numeric.")
        if float(top_level_best) != float(best_val):
            raise CheckpointValidationError(
                "Checkpoint best_val disagrees with resume_state.early_stopping.best_val."
            )
    return resume_state


def is_exact_resumable(data: Mapping[str, Any]) -> bool:
    try:
        validate_exact_resume_checkpoint(data)
    except CheckpointError:
        return False
    return True


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> None:
    """Move nested optimizer tensor state to the parameter device after CPU loading."""
    target = torch.device(device)
    for parameter_state in optimizer.state.values():
        for key, value in list(parameter_state.items()):
            parameter_state[key] = _move_value_to_device(value, target)


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
        # An optimizer without all current resume fields is ambiguous and fails closed.
        kind = None
    else:
        # Anchor/SFT and bare state-dict containers remain inference-compatible.
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

    if operation is CheckpointOperation.PRETRAIN_EXACT_RESUME:
        validate_exact_resume_checkpoint(data)


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


def _validated_rng_state(
    state: Mapping[str, Any],
) -> tuple[tuple[Any, ...], torch.Tensor, list[torch.Tensor] | None]:
    if not isinstance(state, Mapping):
        raise CheckpointValidationError("resume_state.rng_state must be a mapping.")
    missing = sorted({"python", "torch_cpu", "torch_cuda"} - set(state))
    if missing:
        raise CheckpointValidationError(
            "resume_state.rng_state is incomplete; missing: " + ", ".join(missing) + "."
        )
    python_state = state["python"]
    if not isinstance(python_state, tuple):
        raise CheckpointValidationError(
            "resume_state.rng_state.python must be a Python random-state tuple."
        )
    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            "resume_state.rng_state.python is not a valid Python random state."
        ) from exc
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype is not torch.uint8:
        raise CheckpointValidationError(
            "resume_state.rng_state.torch_cpu must be a uint8 tensor."
        )
    try:
        torch.Generator(device="cpu").set_state(cpu_state.cpu())
    except RuntimeError as exc:
        raise CheckpointValidationError(
            "resume_state.rng_state.torch_cpu is not a valid PyTorch CPU RNG state."
        ) from exc
    raw_cuda_states = state["torch_cuda"]
    if raw_cuda_states is None:
        cuda_states = None
    elif isinstance(raw_cuda_states, (list, tuple)) and all(
        isinstance(value, torch.Tensor) and value.dtype is torch.uint8
        for value in raw_cuda_states
    ):
        cuda_states = list(raw_cuda_states)
    else:
        raise CheckpointValidationError(
            "resume_state.rng_state.torch_cuda must be null or a sequence of uint8 tensors."
        )
    return python_state, cpu_state, cuda_states


def _validated_scaler_state(state: Mapping[str, Any]) -> tuple[bool, Mapping[str, Any]]:
    if not isinstance(state, Mapping):
        raise CheckpointValidationError("resume_state.amp_scaler must be a mapping.")
    missing = sorted({"enabled", "state"} - set(state))
    if missing:
        raise CheckpointValidationError(
            "resume_state.amp_scaler is incomplete; missing: " + ", ".join(missing) + "."
        )
    enabled = state["enabled"]
    saved_state = state["state"]
    if not isinstance(enabled, bool):
        raise CheckpointValidationError("resume_state.amp_scaler.enabled must be a boolean.")
    if not isinstance(saved_state, Mapping):
        raise CheckpointValidationError("resume_state.amp_scaler.state must be a mapping.")
    if enabled and not saved_state:
        raise CheckpointValidationError(
            "An enabled AMP scaler requires a non-empty saved state."
        )
    return enabled, saved_state


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    return value


def _safe_value(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _normalized_path(path: Path) -> str:
    return path.as_posix().casefold()


def _selection_key(
    checkpoint: LoadedCheckpoint,
    operation: CheckpointOperation,
) -> tuple[int, str]:
    normalized_path = _normalized_path(checkpoint.path)
    if operation in {
        CheckpointOperation.PRETRAIN_RESUME,
        CheckpointOperation.PRETRAIN_EXACT_RESUME,
    }:
        # Progress is comparable; normalized path is the equal-progress tie-breaker.
        return checkpoint.info.progress or 0, normalized_path
    # Iterations are not comparable across pretraining and stage-local SFT/model loading.
    return checkpoint.path.stat().st_mtime_ns, normalized_path
