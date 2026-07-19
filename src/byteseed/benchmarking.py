"""Versioned generation benchmark measurement and report helpers."""

from __future__ import annotations

import json
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import torch

from .evaluation import EvaluationValidationError, isolated_rng
from .provenance import canonical_sha256


BENCHMARK_REPORT_VERSION = 3
GENERATION_BENCHMARK_VERSION = 3
SUPPORTED_BENCHMARK_REPORT_VERSIONS = (1, 2, 3)
BENCHMARK_REPORT_KIND = "generation_benchmark"


class BenchmarkValidationError(ValueError):
    """Raised when benchmark inputs or reports violate the versioned contract."""


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 1337
    warmup_runs: int = 2
    measured_runs: int = 10
    temperature: float = 0.2
    top_k: int | None = 5
    max_new_tokens: int = 80
    repetition_penalty: float = 1.0
    stop_token_ids: tuple[int, ...] = ()
    device: str = "cpu"
    dtype: str = "fp32"
    compile: bool = False
    deterministic_algorithms: bool = False
    prompt_format_version: int = 1
    prompt_id: str = "benchmark.default-prompt"
    prompt_digest: str = ""
    input_token_count: int = 0
    attention_backend: str = "manual"
    kv_cache: bool = False

    def validate(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise BenchmarkValidationError("benchmark seed must be an integer")
        for label, value, allow_zero in (
            ("warmup_runs", self.warmup_runs, True),
            ("measured_runs", self.measured_runs, False),
            ("max_new_tokens", self.max_new_tokens, False),
            ("input_token_count", self.input_token_count, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if allow_zero else 1)
            ):
                raise BenchmarkValidationError(f"benchmark {label} is invalid")
        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0:
            raise BenchmarkValidationError("benchmark temperature must be positive")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0
        ):
            raise BenchmarkValidationError("benchmark top_k must be null or positive")
        if self.repetition_penalty <= 0:
            raise BenchmarkValidationError("benchmark repetition_penalty must be positive")
        if len(self.stop_token_ids) != len(set(self.stop_token_ids)) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in self.stop_token_ids
        ):
            raise BenchmarkValidationError("benchmark stop_token_ids are invalid")
        if not self.device.strip() or not self.dtype.strip() or not self.prompt_id.strip():
            raise BenchmarkValidationError(
                "benchmark device, dtype, and prompt_id must be non-empty"
            )
        if self.prompt_format_version != 1:
            raise BenchmarkValidationError(
                f"unsupported benchmark prompt format version: {self.prompt_format_version}"
            )
        if self.prompt_digest and re.fullmatch(r"[0-9a-f]{64}", self.prompt_digest) is None:
            raise BenchmarkValidationError("benchmark prompt_digest must be SHA-256")
        if self.attention_backend not in ("manual", "sdpa"):
            raise BenchmarkValidationError(
                "benchmark attention_backend must be manual or sdpa"
            )
        if not isinstance(self.kv_cache, bool):
            raise BenchmarkValidationError("benchmark kv_cache must be boolean")
        if self.compile and self.kv_cache:
            raise BenchmarkValidationError(
                "benchmark compile and kv_cache cannot both be enabled"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "seed": self.seed,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "temperature": float(self.temperature),
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": float(self.repetition_penalty),
            "stop_token_ids": list(self.stop_token_ids),
            "device": self.device,
            "dtype": self.dtype,
            "compile": self.compile,
            "deterministic_algorithms": self.deterministic_algorithms,
            "prompt_format_version": self.prompt_format_version,
            "prompt_id": self.prompt_id,
            "prompt_digest": self.prompt_digest,
            "input_token_count": self.input_token_count,
            "attention_backend": self.attention_backend,
            "kv_cache": self.kv_cache,
        }


@dataclass(frozen=True)
class BenchmarkRun:
    index: int
    elapsed_seconds: float
    generated_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "elapsed_seconds": self.elapsed_seconds,
            "generated_token_count": self.generated_token_count,
            "tokens_per_second": self.generated_token_count / self.elapsed_seconds,
        }


def measure_generation(
    run_once: Callable[[], int],
    config: BenchmarkConfig,
    *,
    clock: Callable[[], float] = perf_counter,
    synchronize: Callable[[], None] = lambda: None,
    after_warmups: Callable[[], None] = lambda: None,
) -> tuple[BenchmarkRun, ...]:
    """Measure generation with synchronized warm-ups excluded from aggregates."""

    config.validate()
    if not config.prompt_digest:
        raise BenchmarkValidationError(
            "benchmark reports require a stable prompt digest"
        )
    try:
        with isolated_rng(
            config.seed,
            device=config.device,
            stochastic=True,
            deterministic_algorithms=config.deterministic_algorithms,
        ):
            for _ in range(config.warmup_runs):
                synchronize()
                run_once()
                synchronize()
            after_warmups()
            results: list[BenchmarkRun] = []
            for index in range(1, config.measured_runs + 1):
                synchronize()
                start = clock()
                generated_token_count = run_once()
                synchronize()
                elapsed = clock() - start
                if elapsed <= 0:
                    raise BenchmarkValidationError(
                        "benchmark elapsed time must be positive"
                    )
                if (
                    isinstance(generated_token_count, bool)
                    or not isinstance(generated_token_count, int)
                    or generated_token_count < 0
                ):
                    raise BenchmarkValidationError(
                        "benchmark generated token count must be non-negative"
                    )
                results.append(
                    BenchmarkRun(index, float(elapsed), generated_token_count)
                )
    except EvaluationValidationError as exc:
        raise BenchmarkValidationError(str(exc)) from exc
    return tuple(results)


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise BenchmarkValidationError(
            "benchmark identity metadata must be JSON-compatible"
        ) from exc


def build_benchmark_report(
    config: BenchmarkConfig,
    runs: Sequence[BenchmarkRun],
    *,
    checkpoint_identity: Mapping[str, Any] | None = None,
    model_configuration: Mapping[str, Any] | None = None,
    parameter_count: int | None = None,
    tokenizer_identity: Mapping[str, Any] | None = None,
    peak_cuda_memory_mib: float | None = None,
    warnings: Sequence[str] = (),
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config.validate()
    if len(runs) != config.measured_runs:
        raise BenchmarkValidationError(
            "measured benchmark runs do not match configuration"
        )
    total_seconds = sum(run.elapsed_seconds for run in runs)
    total_tokens = sum(run.generated_token_count for run in runs)
    if any(run.elapsed_seconds <= 0 for run in runs) or total_seconds <= 0:
        raise BenchmarkValidationError("benchmark elapsed time must be positive")
    if config.device.casefold().startswith("cuda"):
        if peak_cuda_memory_mib is None or peak_cuda_memory_mib < 0:
            raise BenchmarkValidationError(
                "CUDA benchmark reports require valid peak memory"
            )
    elif peak_cuda_memory_mib is not None:
        raise BenchmarkValidationError(
            "CPU benchmark reports must not claim CUDA peak memory"
        )
    report: dict[str, Any] = {
        "version": BENCHMARK_REPORT_VERSION,
        "kind": BENCHMARK_REPORT_KIND,
        "algorithm": "sha256",
        "benchmark_version": GENERATION_BENCHMARK_VERSION,
        "configuration": config.to_dict(),
        "checkpoint": _json_mapping(checkpoint_identity),
        "model_configuration": _json_mapping(model_configuration) or {},
        "parameter_count": parameter_count,
        "tokenizer_identity": _json_mapping(tokenizer_identity),
        "environment": _json_mapping(environment)
        or {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "cuda_version": torch.version.cuda,
            "device": config.device,
            "dtype": config.dtype,
            "compile": config.compile,
            "deterministic_algorithms_enabled": config.deterministic_algorithms,
        },
        "measurements": {
            "runs": [run.to_dict() for run in runs],
            "aggregate": {
                "total_elapsed_seconds": total_seconds,
                "mean_elapsed_seconds": total_seconds / len(runs),
                "minimum_elapsed_seconds": min(run.elapsed_seconds for run in runs),
                "maximum_elapsed_seconds": max(run.elapsed_seconds for run in runs),
                "total_generated_tokens": total_tokens,
                "tokens_per_second": total_tokens / total_seconds,
            },
            "peak_cuda_memory_mib": peak_cuda_memory_mib,
        },
        "warnings": list(warnings),
    }
    report["digest"] = canonical_sha256(report)
    validate_benchmark_report(report)
    return report


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkValidationError(f"{label} must be a mapping")
    return value


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def validate_benchmark_report(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise BenchmarkValidationError("benchmark report must be a mapping")
    if set(report) != {
        "version",
        "kind",
        "algorithm",
        "benchmark_version",
        "configuration",
        "checkpoint",
        "model_configuration",
        "parameter_count",
        "tokenizer_identity",
        "environment",
        "measurements",
        "warnings",
        "digest",
    }:
        raise BenchmarkValidationError(
            "benchmark report root fields are missing or malformed"
        )
    version = report.get("version")
    if version not in SUPPORTED_BENCHMARK_REPORT_VERSIONS:
        qualifier = (
            "future "
            if isinstance(version, int) and version > BENCHMARK_REPORT_VERSION
            else ""
        )
        raise BenchmarkValidationError(
            f"unsupported {qualifier}benchmark report version: {version!r}"
        )
    if report.get("kind") != BENCHMARK_REPORT_KIND:
        raise BenchmarkValidationError(
            f"wrong benchmark report kind: {report.get('kind')!r}"
        )
    if report.get("algorithm") != "sha256":
        raise BenchmarkValidationError("benchmark report algorithm must be sha256")
    expected_benchmark_version = version
    if report.get("benchmark_version") != expected_benchmark_version:
        raise BenchmarkValidationError(
            f"unsupported generation benchmark version: "
            f"{report.get('benchmark_version')!r}"
        )

    configuration = _mapping(
        report.get("configuration"), "benchmark configuration"
    )
    required_configuration = {
        "seed", "warmup_runs", "measured_runs", "temperature", "top_k",
        "max_new_tokens", "repetition_penalty", "stop_token_ids", "device",
        "dtype", "compile", "deterministic_algorithms", "prompt_format_version",
        "prompt_id", "prompt_digest", "input_token_count",
    }
    if version >= 2:
        required_configuration.add("attention_backend")
    if version >= 3:
        required_configuration.add("kv_cache")
    if set(configuration) != required_configuration:
        raise BenchmarkValidationError(
            "benchmark configuration is missing or malformed"
        )
    stop_ids = configuration["stop_token_ids"]
    if not isinstance(stop_ids, list):
        raise BenchmarkValidationError("benchmark stop_token_ids must be a list")
    config = BenchmarkConfig(
        seed=configuration["seed"],
        warmup_runs=configuration["warmup_runs"],
        measured_runs=configuration["measured_runs"],
        temperature=configuration["temperature"],
        top_k=configuration["top_k"],
        max_new_tokens=configuration["max_new_tokens"],
        repetition_penalty=configuration["repetition_penalty"],
        stop_token_ids=tuple(stop_ids),
        device=configuration["device"],
        dtype=configuration["dtype"],
        compile=configuration["compile"],
        deterministic_algorithms=configuration["deterministic_algorithms"],
        prompt_format_version=configuration["prompt_format_version"],
        prompt_id=configuration["prompt_id"],
        prompt_digest=configuration["prompt_digest"],
        input_token_count=configuration["input_token_count"],
        attention_backend=configuration.get("attention_backend", "manual"),
        kv_cache=configuration.get("kv_cache", False),
    )
    if not isinstance(config.kv_cache, bool):
        raise BenchmarkValidationError(
            "benchmark kv_cache configuration field must be boolean"
        )
    if not isinstance(config.compile, bool) or not isinstance(
        config.deterministic_algorithms, bool
    ):
        raise BenchmarkValidationError(
            "benchmark boolean configuration fields are malformed"
        )
    config.validate()
    if not config.prompt_digest:
        raise BenchmarkValidationError(
            "benchmark reports require a stable prompt digest"
        )

    checkpoint = report.get("checkpoint")
    if checkpoint is not None:
        checkpoint = _mapping(checkpoint, "benchmark checkpoint identity")
        if not isinstance(checkpoint.get("logical_name"), str) or not checkpoint[
            "logical_name"
        ].strip():
            raise BenchmarkValidationError(
                "benchmark checkpoint logical_name must be non-empty text"
            )
        artifact_digest = checkpoint.get("artifact_sha256")
        if artifact_digest is not None and (
            not isinstance(artifact_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact_digest) is None
        ):
            raise BenchmarkValidationError(
                "benchmark checkpoint artifact_sha256 must be lowercase SHA-256"
            )
    model_configuration = _mapping(
        report.get("model_configuration"), "benchmark model configuration"
    )
    tokenizer_identity = report.get("tokenizer_identity")
    if tokenizer_identity is not None:
        _mapping(tokenizer_identity, "benchmark tokenizer identity")
    if any(
        _contains_absolute_path(value)
        for value in (checkpoint, model_configuration, tokenizer_identity)
    ):
        raise BenchmarkValidationError(
            "benchmark identity metadata must not contain absolute paths"
        )
    parameter_count = report.get("parameter_count")
    if parameter_count is not None and (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count < 0
    ):
        raise BenchmarkValidationError(
            "benchmark parameter_count must be null or non-negative"
        )
    environment = _mapping(report.get("environment"), "benchmark environment")
    for field in ("python_version", "pytorch_version", "device", "dtype"):
        if not isinstance(environment.get(field), str) or not environment[field].strip():
            raise BenchmarkValidationError(
                f"benchmark environment {field} must be non-empty text"
            )
    for field in ("compile", "deterministic_algorithms_enabled"):
        if not isinstance(environment.get(field), bool):
            raise BenchmarkValidationError(
                f"benchmark environment {field} must be boolean"
            )
    if (
        environment["device"] != config.device
        or environment["dtype"] != config.dtype
        or environment["compile"] is not config.compile
        or environment["deterministic_algorithms_enabled"]
        is not config.deterministic_algorithms
    ):
        raise BenchmarkValidationError(
            "benchmark environment and configuration fields disagree"
        )

    measurements = _mapping(
        report.get("measurements"), "benchmark measurements"
    )
    if set(measurements) != {"runs", "aggregate", "peak_cuda_memory_mib"}:
        raise BenchmarkValidationError("benchmark measurement fields are malformed")
    runs = measurements.get("runs")
    if not isinstance(runs, list) or len(runs) != config.measured_runs:
        raise BenchmarkValidationError(
            "benchmark measured runs do not match configuration"
        )
    elapsed_values: list[float] = []
    token_values: list[int] = []
    for expected_index, value in enumerate(runs, start=1):
        run = _mapping(value, "benchmark run")
        if set(run) != {
            "index",
            "elapsed_seconds",
            "generated_token_count",
            "tokens_per_second",
        }:
            raise BenchmarkValidationError("benchmark run fields are malformed")
        if run.get("index") != expected_index:
            raise BenchmarkValidationError(
                "benchmark run ordering is not deterministic"
            )
        elapsed = run.get("elapsed_seconds")
        tokens = run.get("generated_token_count")
        throughput = run.get("tokens_per_second")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or elapsed <= 0
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 0
            or not isinstance(throughput, (int, float))
            or not math.isclose(
                float(throughput), tokens / float(elapsed), rel_tol=1e-12
            )
        ):
            raise BenchmarkValidationError("benchmark run measurement is malformed")
        elapsed_values.append(float(elapsed))
        token_values.append(tokens)

    aggregate = _mapping(
        measurements.get("aggregate"), "benchmark aggregate"
    )
    if set(aggregate) != {
        "total_elapsed_seconds",
        "mean_elapsed_seconds",
        "minimum_elapsed_seconds",
        "maximum_elapsed_seconds",
        "total_generated_tokens",
        "tokens_per_second",
    }:
        raise BenchmarkValidationError("benchmark aggregate fields are malformed")
    total_elapsed = sum(elapsed_values)
    total_tokens = sum(token_values)
    expected_values = {
        "total_elapsed_seconds": total_elapsed,
        "mean_elapsed_seconds": total_elapsed / len(elapsed_values),
        "minimum_elapsed_seconds": min(elapsed_values),
        "maximum_elapsed_seconds": max(elapsed_values),
        "tokens_per_second": total_tokens / total_elapsed,
    }
    if aggregate.get("total_generated_tokens") != total_tokens or any(
        not isinstance(aggregate.get(field), (int, float))
        or not math.isclose(
            float(aggregate[field]), expected, rel_tol=1e-12
        )
        for field, expected in expected_values.items()
    ):
        raise BenchmarkValidationError(
            "benchmark aggregate does not match measured runs"
        )
    peak_memory = measurements.get("peak_cuda_memory_mib")
    if config.device.casefold().startswith("cuda"):
        if (
            isinstance(peak_memory, bool)
            or not isinstance(peak_memory, (int, float))
            or peak_memory < 0
        ):
            raise BenchmarkValidationError(
                "CUDA benchmark peak-memory measurement is invalid"
            )
    elif peak_memory is not None:
        raise BenchmarkValidationError(
            "CPU benchmark report must use null CUDA peak memory"
        )
    if not isinstance(report.get("warnings"), list) or not all(
        isinstance(item, str) for item in report["warnings"]
    ):
        raise BenchmarkValidationError("benchmark warnings must be a text list")

    digest = report.get("digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise BenchmarkValidationError(
            "benchmark report digest must be lowercase SHA-256"
        )
    payload = {key: value for key, value in report.items() if key != "digest"}
    if canonical_sha256(payload) != digest:
        raise BenchmarkValidationError("benchmark report digest mismatch")


def load_benchmark_report(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_benchmark_report(value)
    return value


def write_benchmark_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    validate_benchmark_report(report)
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"benchmark report already exists: {destination.name}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_benchmark_report(report: Mapping[str, Any]) -> str:
    validate_benchmark_report(report)
    config = report["configuration"]
    aggregate = report["measurements"]["aggregate"]
    lines = [
        "ByteSeed generation benchmark",
        f"Device: {config['device']}",
        f"Dtype: {config['dtype']}",
        f"Attention backend: {config.get('attention_backend', 'manual')}",
        f"KV cache: {'on' if config.get('kv_cache', False) else 'off'}",
        f"Warm-up runs: {config['warmup_runs']} (excluded)",
        f"Measured runs: {config['measured_runs']}",
        f"Mean latency: {aggregate['mean_elapsed_seconds']:.4f} s",
        f"Throughput: {aggregate['tokens_per_second']:.2f} tokens/s",
    ]
    peak = report["measurements"]["peak_cuda_memory_mib"]
    lines.append(
        "Peak CUDA memory: n/a"
        if peak is None
        else f"Peak CUDA memory: {peak:.1f} MiB"
    )
    for warning in report["warnings"]:
        lines.append(f"Warning: {warning}")
    lines.append(
        "Timing measurements are environment-dependent and are not comparable "
        "across unlike systems."
    )
    lines.append(f"Report digest: {report['digest']}")
    return "\n".join(lines)
