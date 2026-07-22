"""Deterministic ByteSeed evaluation, scoring, provenance, and report helpers."""

from __future__ import annotations

import json
import platform
import random
import re
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

from .checkpoint import capture_rng_state, restore_rng_state
from .data_quality import validate_data_quality_report
from .eval_prompts import (
    ANCHOR_RETENTION_SUITE,
    RUBRIC_VERSION,
    EvaluationPrompt,
    EvaluationSuite,
    get_evaluation_suite,
    serialize_evaluation_suite,
    validate_evaluation_suite,
)
from .provenance import canonical_sha256, validate_data_manifest


EVALUATION_REPORT_VERSION = 1
EVALUATION_RUNNER_VERSION = 1
EVALUATION_REPORT_KIND = "evaluation"
CONTAMINATION_STATUSES = frozenset(
    {
        "verified_clean",
        "contaminated",
        "audit_unavailable",
        "audit_does_not_cover_suite",
        "provenance_mismatch",
        "legacy_unverified",
    }
)


class EvaluationValidationError(ValueError):
    """Raised when evaluation inputs or reports violate the versioned contract."""


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 1337
    temperature: float = 0.2
    top_k: int | None = 5
    max_new_tokens: int = 80
    repetition_penalty: float = 1.0
    stop_token_ids: tuple[int, ...] = ()
    stop_at_end: bool = True
    dtype: str = "fp32"
    device: str = "cpu"
    compile: bool = False
    batch_size: int = 1
    prompt_format_version: int = 1
    deterministic_algorithms: bool = False
    sampling_mode: str = "stochastic"

    def validate(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EvaluationValidationError("generation seed must be an integer")
        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0:
            raise EvaluationValidationError("generation temperature must be positive")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0
        ):
            raise EvaluationValidationError("generation top_k must be null or a positive integer")
        if self.max_new_tokens <= 0:
            raise EvaluationValidationError("generation max_new_tokens must be positive")
        if self.repetition_penalty <= 0:
            raise EvaluationValidationError("generation repetition_penalty must be positive")
        if len(self.stop_token_ids) != len(set(self.stop_token_ids)) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in self.stop_token_ids
        ):
            raise EvaluationValidationError(
                "generation stop_token_ids must be unique non-negative integers"
            )
        if not self.dtype.strip() or not self.device.strip():
            raise EvaluationValidationError("generation dtype and device must be non-empty")
        if self.batch_size <= 0:
            raise EvaluationValidationError("generation batch_size must be positive")
        if self.prompt_format_version != 1:
            raise EvaluationValidationError(
                f"unsupported prompt format version: {self.prompt_format_version}"
            )
        if self.sampling_mode not in {"stochastic", "greedy"}:
            raise EvaluationValidationError(
                "generation sampling_mode must be 'stochastic' or 'greedy'"
            )
        if self.sampling_mode == "greedy" and self.top_k != 1:
            raise EvaluationValidationError("greedy evaluation requires top_k=1")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "seed": self.seed,
            "temperature": float(self.temperature),
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": float(self.repetition_penalty),
            "stop_token_ids": list(self.stop_token_ids),
            "stop_at_end": self.stop_at_end,
            "dtype": self.dtype,
            "device": self.device,
            "compile": self.compile,
            "batch_size": self.batch_size,
            "prompt_format_version": self.prompt_format_version,
            "deterministic_algorithms": self.deterministic_algorithms,
            "sampling_mode": self.sampling_mode,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GenerationConfig:
        required = {
            "seed", "temperature", "top_k", "max_new_tokens", "repetition_penalty",
            "stop_token_ids", "stop_at_end", "dtype", "device", "compile",
            "batch_size", "prompt_format_version", "deterministic_algorithms",
            "sampling_mode",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise EvaluationValidationError(
                "evaluation report generation configuration is missing or malformed"
            )
        stop_ids = value["stop_token_ids"]
        if not isinstance(stop_ids, list):
            raise EvaluationValidationError("generation stop_token_ids must be a list")
        config = cls(
            seed=value["seed"],
            temperature=value["temperature"],
            top_k=value["top_k"],
            max_new_tokens=value["max_new_tokens"],
            repetition_penalty=value["repetition_penalty"],
            stop_token_ids=tuple(stop_ids),
            stop_at_end=value["stop_at_end"],
            dtype=value["dtype"],
            device=value["device"],
            compile=value["compile"],
            batch_size=value["batch_size"],
            prompt_format_version=value["prompt_format_version"],
            deterministic_algorithms=value["deterministic_algorithms"],
            sampling_mode=value["sampling_mode"],
        )
        if not all(
            isinstance(flag, bool)
            for flag in (config.stop_at_end, config.compile, config.deterministic_algorithms)
        ):
            raise EvaluationValidationError("generation boolean settings are malformed")
        config.validate()
        return config


@dataclass(frozen=True)
class GeneratedCaseOutput:
    response: str
    generated_token_count: int
    stop_reason: str
    warnings: tuple[str, ...] = ()


EvaluationBatchGenerator = Callable[
    [tuple[EvaluationPrompt, ...], GenerationConfig],
    Sequence[GeneratedCaseOutput],
]


def normalize_rubric_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def score_response(case: EvaluationPrompt, response: str) -> dict[str, Any]:
    rubric = case.rubric
    if rubric is None:
        return {
            "version": RUBRIC_VERSION,
            "status": "unscored",
            "passed": None,
            "matched_required_concepts": [],
            "missing_required_concepts": [],
            "forbidden_matches": [],
            "human_review_required": True,
        }
    if rubric.version != RUBRIC_VERSION:
        raise EvaluationValidationError(
            f"unsupported scoring/rubric version: {rubric.version}"
        )

    answer = normalize_rubric_text(response)
    matched: list[str] = []
    missing: list[str] = []
    for requirement in rubric.required:
        alternatives = tuple(normalize_rubric_text(item) for item in requirement.accepted_phrases)
        if any(phrase and phrase in answer for phrase in alternatives):
            matched.append(requirement.concept)
        else:
            missing.append(requirement.concept)

    forbidden = [
        term for term in rubric.forbidden_terms if normalize_rubric_text(term) in answer
    ]
    for combination in rubric.forbidden_combinations:
        normalized = tuple(normalize_rubric_text(term) for term in combination)
        if all(term in answer for term in normalized):
            forbidden.append(" + ".join(combination))

    if rubric.human_review:
        status = "human-review"
        passed: bool | None = None
    else:
        passed = bool(answer) and not missing and not forbidden
        status = "pass" if passed else "fail"
    return {
        "version": rubric.version,
        "status": status,
        "passed": passed,
        "matched_required_concepts": matched,
        "missing_required_concepts": missing,
        "forbidden_matches": forbidden,
        "human_review_required": rubric.human_review,
    }


@contextmanager
def isolated_rng(
    seed: int,
    *,
    device: str = "cpu",
    stochastic: bool = True,
    deterministic_algorithms: bool = False,
) -> Iterator[None]:
    """Apply evaluation RNG policy and finally restore the caller's state."""

    state = capture_rng_state()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic_algorithms)
        if stochastic:
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if device.casefold().startswith("cuda"):
                if not torch.cuda.is_available() or not torch.cuda.is_initialized():
                    raise EvaluationValidationError(
                        "CUDA evaluation requires CUDA to be available and already initialized"
                    )
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        try:
            restore_rng_state(state)
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)


def _contamination_result(
    status: str,
    *,
    suite_covered: bool,
    quality_report_digest: str | None,
    data_manifest_digest: str | None,
    matching_prompt_ids: Sequence[str] = (),
    override_used: bool = False,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    held_out_status = (
        "verified-clean"
        if status == "verified_clean"
        else "contaminated"
        if status == "contaminated"
        else "unverified"
    )
    return {
        "status": status,
        "held_out_status": held_out_status,
        "suite_covered": suite_covered,
        "quality_report_digest": quality_report_digest,
        "data_manifest_digest": data_manifest_digest,
        "matching_prompt_ids": list(matching_prompt_ids),
        "override_used": override_used,
        "warnings": list(warnings),
    }


def classify_contamination(
    suite: EvaluationSuite,
    *,
    quality_report: Mapping[str, Any] | None = None,
    data_manifest: Mapping[str, Any] | None = None,
    checkpoint_data_manifest_digest: str | None = None,
) -> dict[str, Any]:
    """Classify exact suite coverage against linked PR 6 provenance evidence."""

    validate_evaluation_suite(suite)
    if quality_report is not None:
        try:
            validate_data_quality_report(quality_report)
        except Exception as exc:
            raise EvaluationValidationError(f"invalid data-quality report: {exc}") from exc
    if suite.suite_id == ANCHOR_RETENTION_SUITE:
        return _contamination_result(
            "contaminated",
            suite_covered=True,
            quality_report_digest=(quality_report or {}).get("digest"),
            data_manifest_digest=(data_manifest or {}).get("digest"),
            matching_prompt_ids=[case.prompt_id for case in suite.cases],
            warnings=(
                "Historical Anchor prompts are known to overlap historical SFT data; "
                "this suite is retention-only.",
            ),
        )

    if quality_report is None:
        return _contamination_result(
            "audit_unavailable",
            suite_covered=False,
            quality_report_digest=None,
            data_manifest_digest=(data_manifest or {}).get("digest"),
            warnings=(
                "No compatible data-quality report was supplied; held-out status is unverified.",
            ),
        )
    report_digest = quality_report["digest"]
    findings = quality_report.get("contamination_findings", [])
    suite_prompt_ids = [case.prompt_id for case in suite.cases]
    matching_ids = [
        prompt_id
        for prompt_id in suite_prompt_ids
        if any(
            finding.get("suite") == suite.suite_id
            and finding.get("prompt_id") == prompt_id
            for finding in findings
        )
    ]
    override_used = bool(
        quality_report.get("policy", {}).get("allow_eval_contamination", False)
    )
    if matching_ids:
        warnings = ["The data-quality audit found exact normalized overlap for this suite."]
        if override_used:
            warnings.append("The data-quality contamination override was enabled.")
        return _contamination_result(
            "contaminated",
            suite_covered=False,
            quality_report_digest=report_digest,
            data_manifest_digest=(data_manifest or {}).get("digest"),
            matching_prompt_ids=matching_ids,
            override_used=override_used,
            warnings=warnings,
        )

    audit = quality_report.get("evaluation_audit")
    coverage: Mapping[str, Any] | None = None
    if isinstance(audit, Mapping) and isinstance(audit.get("suites"), list):
        for value in audit["suites"]:
            if isinstance(value, Mapping) and value.get("suite_id") == suite.suite_id:
                coverage = value
                break
    covered_ids = coverage.get("prompt_ids") if coverage is not None else None
    suite_covered = (
        isinstance(audit, Mapping)
        and audit.get("registry_version") == 1
        and coverage is not None
        and coverage.get("suite_version") == suite.version
        and covered_ids == suite_prompt_ids
    )
    if not suite_covered:
        return _contamination_result(
            "audit_does_not_cover_suite",
            suite_covered=False,
            quality_report_digest=report_digest,
            data_manifest_digest=(data_manifest or {}).get("digest"),
            override_used=override_used,
            warnings=(
                "The data-quality audit does not record exact coverage for this suite; "
                "held-out status is unverified.",
            ),
        )

    if data_manifest is None:
        return _contamination_result(
            "provenance_mismatch",
            suite_covered=True,
            quality_report_digest=report_digest,
            data_manifest_digest=None,
            override_used=override_used,
            warnings=(
                "Exact suite audit coverage is present, but its document-aware v2 data "
                "manifest is unavailable.",
            ),
        )
    try:
        validate_data_manifest(data_manifest)
    except Exception as exc:
        raise EvaluationValidationError(f"invalid data manifest: {exc}") from exc

    manifest_digest = data_manifest["digest"]
    if data_manifest.get("version") != 2:
        return _contamination_result(
            "legacy_unverified",
            suite_covered=True,
            quality_report_digest=report_digest,
            data_manifest_digest=manifest_digest,
            override_used=override_used,
            warnings=("Legacy data provenance cannot verify candidate held-out status.",),
        )
    linked_report_digest = data_manifest["preprocessing"]["data_quality"]["report_digest"]
    if (
        linked_report_digest != report_digest
        or checkpoint_data_manifest_digest is None
        or checkpoint_data_manifest_digest != manifest_digest
    ):
        return _contamination_result(
            "provenance_mismatch",
            suite_covered=True,
            quality_report_digest=report_digest,
            data_manifest_digest=manifest_digest,
            override_used=override_used,
            warnings=(
                "Suite audit, data manifest, and checkpoint data identity do not all match; "
                "held-out status is unverified.",
            ),
        )
    if override_used:
        return _contamination_result(
            "contaminated",
            suite_covered=True,
            quality_report_digest=report_digest,
            data_manifest_digest=manifest_digest,
            override_used=True,
            warnings=(
                "The data-quality contamination override was enabled; candidate held-out "
                "wording is withheld.",
            ),
        )
    return _contamination_result(
        "verified_clean",
        suite_covered=True,
        quality_report_digest=report_digest,
        data_manifest_digest=manifest_digest,
    )


def runtime_environment(config: GenerationConfig) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cuda_version": torch.version.cuda,
        "device": config.device,
        "dtype": config.dtype,
        "compile": config.compile,
        "deterministic_algorithms_enabled": config.deterministic_algorithms,
    }


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise EvaluationValidationError("report identity metadata must be JSON-compatible") from exc


def _metric_label(suite: EvaluationSuite, contamination: Mapping[str, Any]) -> str:
    if suite.suite_id == ANCHOR_RETENTION_SUITE:
        return "Anchor-retention regression"
    if contamination["status"] == "verified_clean":
        return "Candidate held-out paraphrase checks"
    return "Candidate paraphrase checks"


def run_evaluation(
    suite: EvaluationSuite,
    generation: GenerationConfig,
    generator: EvaluationBatchGenerator,
    *,
    checkpoint_identity: Mapping[str, Any] | None = None,
    model_configuration: Mapping[str, Any] | None = None,
    parameter_count: int | None = None,
    tokenizer_identity: Mapping[str, Any] | None = None,
    data_manifest_digest: str | None = None,
    quality_report: Mapping[str, Any] | None = None,
    data_manifest: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a suite in stable order and return a validated report-v1 mapping."""

    validate_evaluation_suite(suite)
    generation.validate()
    if parameter_count is not None and (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count < 0
    ):
        raise EvaluationValidationError("parameter_count must be null or non-negative")
    checkpoint = _json_mapping(checkpoint_identity)
    checkpoint_manifest_digest = data_manifest_digest
    if checkpoint_manifest_digest is None and checkpoint is not None:
        value = checkpoint.get("data_manifest_digest")
        checkpoint_manifest_digest = value if isinstance(value, str) else None
    contamination = classify_contamination(
        suite,
        quality_report=quality_report,
        data_manifest=data_manifest,
        checkpoint_data_manifest_digest=checkpoint_manifest_digest,
    )

    outputs: list[GeneratedCaseOutput] = []
    with isolated_rng(
        generation.seed,
        device=generation.device,
        stochastic=generation.sampling_mode == "stochastic",
        deterministic_algorithms=generation.deterministic_algorithms,
    ):
        for start in range(0, len(suite.cases), generation.batch_size):
            batch = suite.cases[start : start + generation.batch_size]
            generated = tuple(generator(batch, generation))
            if len(generated) != len(batch):
                raise EvaluationValidationError(
                    "evaluation generator returned a different number of outputs than prompts"
                )
            outputs.extend(generated)

    results: list[dict[str, Any]] = []
    for case, output in zip(suite.cases, outputs, strict=True):
        if not isinstance(output.response, str):
            raise EvaluationValidationError("generated response must be text")
        if output.generated_token_count < 0:
            raise EvaluationValidationError("generated token count must be non-negative")
        if not output.stop_reason.strip():
            raise EvaluationValidationError("generation stop reason must be non-empty")
        results.append(
            {
                "prompt_id": case.prompt_id,
                "prompt_text": case.text,
                "prompt_digest": canonical_sha256(
                    {"prompt_id": case.prompt_id, "prompt_text": case.text}
                ),
                "response": output.response,
                "stop_reason": output.stop_reason,
                "generated_token_count": output.generated_token_count,
                "rubric": score_response(case, output.response),
                "warnings": list(output.warnings),
            }
        )

    passed = sum(result["rubric"]["status"] == "pass" for result in results)
    failed = sum(result["rubric"]["status"] == "fail" for result in results)
    unscored = len(results) - passed - failed
    held_out_measured = (
        suite.suite_id != ANCHOR_RETENTION_SUITE
        and contamination["status"] == "verified_clean"
    )
    report: dict[str, Any] = {
        "version": EVALUATION_REPORT_VERSION,
        "kind": EVALUATION_REPORT_KIND,
        "algorithm": "sha256",
        "runner": {
            "version": EVALUATION_RUNNER_VERSION,
            "prompt_format_version": generation.prompt_format_version,
        },
        "suite": {
            "suite_id": suite.suite_id,
            "version": suite.version,
            "purpose": suite.purpose,
            "expected_mode": suite.expected_mode,
            "historical_status": suite.historical_status,
            "case_count": len(suite.cases),
            "digest": canonical_sha256(serialize_evaluation_suite(suite)),
        },
        "generation": generation.to_dict(),
        "checkpoint": checkpoint,
        "model_configuration": _json_mapping(model_configuration) or {},
        "parameter_count": parameter_count,
        "tokenizer_identity": _json_mapping(tokenizer_identity),
        "data_manifest_digest": checkpoint_manifest_digest,
        "contamination": contamination,
        "environment": _json_mapping(environment) or runtime_environment(generation),
        "warnings": list(contamination["warnings"]),
        "results": results,
        "summary": {
            "metric_label": _metric_label(suite, contamination),
            "total_cases": len(results),
            "passed": passed,
            "failed": failed,
            "unscored": unscored,
            "held_out_generalization_measured": held_out_measured,
        },
    }
    report["digest"] = canonical_sha256(report)
    validate_evaluation_report(report)
    return report


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationValidationError(f"{label} must be a mapping")
    return value


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _validate_optional_digest(value: Any, label: str) -> None:
    if value is not None and (
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise EvaluationValidationError(f"{label} must be null or lowercase SHA-256")


def validate_evaluation_report(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise EvaluationValidationError("evaluation report must be a mapping")
    required_root = {
        "version",
        "kind",
        "algorithm",
        "runner",
        "suite",
        "generation",
        "checkpoint",
        "model_configuration",
        "parameter_count",
        "tokenizer_identity",
        "data_manifest_digest",
        "contamination",
        "environment",
        "warnings",
        "results",
        "summary",
        "digest",
    }
    if set(report) != required_root:
        raise EvaluationValidationError(
            "evaluation report root fields are missing or malformed"
        )
    version = report.get("version")
    if version != EVALUATION_REPORT_VERSION:
        qualifier = (
            "future "
            if isinstance(version, int) and version > EVALUATION_REPORT_VERSION
            else ""
        )
        raise EvaluationValidationError(
            f"unsupported {qualifier}evaluation report version: {version!r}"
        )
    if report.get("kind") != EVALUATION_REPORT_KIND:
        raise EvaluationValidationError(
            f"wrong evaluation report kind: {report.get('kind')!r}"
        )
    if report.get("algorithm") != "sha256":
        raise EvaluationValidationError("evaluation report algorithm must be sha256")

    runner = _require_mapping(report.get("runner"), "evaluation report runner")
    if set(runner) != {"version", "prompt_format_version"}:
        raise EvaluationValidationError("evaluation report runner metadata is malformed")
    if runner.get("version") != EVALUATION_RUNNER_VERSION:
        raise EvaluationValidationError(
            f"unsupported evaluation runner version: {runner.get('version')!r}"
        )
    suite = _require_mapping(report.get("suite"), "evaluation report suite")
    if set(suite) != {
        "suite_id",
        "version",
        "purpose",
        "expected_mode",
        "historical_status",
        "case_count",
        "digest",
    }:
        raise EvaluationValidationError("evaluation report suite metadata is malformed")
    for field in ("suite_id", "purpose", "expected_mode", "historical_status"):
        if not isinstance(suite.get(field), str) or not suite[field].strip():
            raise EvaluationValidationError(
                f"evaluation suite {field} must be non-empty text"
            )
    if suite.get("version") != 1:
        raise EvaluationValidationError(
            f"unsupported evaluation suite version: {suite.get('version')!r}"
        )
    if not isinstance(suite.get("digest"), str) or re.fullmatch(
        r"[0-9a-f]{64}", suite["digest"]
    ) is None:
        raise EvaluationValidationError(
            "evaluation suite digest must be lowercase SHA-256"
        )
    try:
        registered_suite = get_evaluation_suite(suite["suite_id"])
    except ValueError as exc:
        raise EvaluationValidationError(str(exc)) from exc
    expected_suite_metadata = {
        "version": registered_suite.version,
        "purpose": registered_suite.purpose,
        "expected_mode": registered_suite.expected_mode,
        "historical_status": registered_suite.historical_status,
        "case_count": len(registered_suite.cases),
        "digest": canonical_sha256(serialize_evaluation_suite(registered_suite)),
    }
    if any(suite.get(field) != value for field, value in expected_suite_metadata.items()):
        raise EvaluationValidationError(
            "evaluation report suite metadata does not match the registered suite identity"
        )
    case_count = suite.get("case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count < 1:
        raise EvaluationValidationError("evaluation suite case_count must be positive")
    generation = GenerationConfig.from_mapping(
        _require_mapping(report.get("generation"), "evaluation report generation")
    )
    if runner.get("prompt_format_version") != generation.prompt_format_version:
        raise EvaluationValidationError(
            "evaluation runner and generation prompt format versions differ"
        )
    checkpoint = report.get("checkpoint")
    if checkpoint is not None:
        checkpoint = _require_mapping(checkpoint, "evaluation checkpoint identity")
        logical_name = checkpoint.get("logical_name")
        if not isinstance(logical_name, str) or not logical_name.strip():
            raise EvaluationValidationError(
                "evaluation checkpoint logical_name must be non-empty text"
            )
        if _contains_absolute_path(checkpoint):
            raise EvaluationValidationError(
                "evaluation checkpoint identity must not contain absolute paths"
            )
        _validate_optional_digest(
            checkpoint.get("artifact_sha256"),
            "evaluation checkpoint artifact_sha256",
        )
    model_configuration = _require_mapping(
        report.get("model_configuration"), "evaluation model configuration"
    )
    tokenizer_identity = report.get("tokenizer_identity")
    if tokenizer_identity is not None:
        _require_mapping(tokenizer_identity, "evaluation tokenizer identity")
    if _contains_absolute_path(model_configuration) or _contains_absolute_path(
        tokenizer_identity
    ):
        raise EvaluationValidationError(
            "evaluation identity metadata must not contain absolute paths"
        )
    parameter_count = report.get("parameter_count")
    if parameter_count is not None and (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count < 0
    ):
        raise EvaluationValidationError(
            "evaluation parameter_count must be null or non-negative"
        )
    _validate_optional_digest(
        report.get("data_manifest_digest"), "evaluation data_manifest_digest"
    )
    environment = _require_mapping(
        report.get("environment"), "evaluation runtime environment"
    )
    for field in ("python_version", "pytorch_version", "device", "dtype"):
        if not isinstance(environment.get(field), str) or not environment[field].strip():
            raise EvaluationValidationError(
                f"evaluation environment {field} must be non-empty text"
            )
    for field in ("compile", "deterministic_algorithms_enabled"):
        if not isinstance(environment.get(field), bool):
            raise EvaluationValidationError(
                f"evaluation environment {field} must be boolean"
            )

    contamination = _require_mapping(
        report.get("contamination"), "evaluation report contamination"
    )
    if set(contamination) != {
        "status",
        "held_out_status",
        "suite_covered",
        "quality_report_digest",
        "data_manifest_digest",
        "matching_prompt_ids",
        "override_used",
        "warnings",
    }:
        raise EvaluationValidationError(
            "evaluation contamination metadata is malformed"
        )
    if contamination.get("status") not in CONTAMINATION_STATUSES:
        raise EvaluationValidationError(
            f"invalid contamination status: {contamination.get('status')!r}"
        )
    if contamination.get("held_out_status") not in {
        "verified-clean",
        "contaminated",
        "unverified",
    }:
        raise EvaluationValidationError("invalid held-out status")
    expected_held_out = (
        "verified-clean"
        if contamination["status"] == "verified_clean"
        else "contaminated"
        if contamination["status"] == "contaminated"
        else "unverified"
    )
    if contamination["held_out_status"] != expected_held_out:
        raise EvaluationValidationError(
            "contamination and held-out status classifications disagree"
        )
    if not isinstance(contamination.get("suite_covered"), bool) or not isinstance(
        contamination.get("override_used"), bool
    ):
        raise EvaluationValidationError(
            "contamination coverage and override fields must be boolean"
        )
    _validate_optional_digest(
        contamination.get("quality_report_digest"),
        "contamination quality_report_digest",
    )
    _validate_optional_digest(
        contamination.get("data_manifest_digest"),
        "contamination data_manifest_digest",
    )
    if not isinstance(contamination.get("matching_prompt_ids"), list) or not all(
        isinstance(item, str) and item.strip()
        for item in contamination["matching_prompt_ids"]
    ):
        raise EvaluationValidationError(
            "contamination matching_prompt_ids must be a text list"
        )
    if not isinstance(contamination.get("warnings"), list) or not all(
        isinstance(item, str) for item in contamination["warnings"]
    ):
        raise EvaluationValidationError("contamination warnings must be a text list")

    results = report.get("results")
    if not isinstance(results, list) or len(results) != case_count:
        raise EvaluationValidationError("evaluation results must match suite case_count")
    prompt_ids: set[str] = set()
    statuses: list[str] = []
    for expected_case, result in zip(registered_suite.cases, results, strict=True):
        item = _require_mapping(result, "evaluation result entry")
        expected_result_fields = {
            "prompt_id",
            "prompt_text",
            "prompt_digest",
            "response",
            "stop_reason",
            "generated_token_count",
            "rubric",
            "warnings",
        }
        if set(item) != expected_result_fields:
            missing = sorted(expected_result_fields - set(item))
            extra = sorted(set(item) - expected_result_fields)
            raise EvaluationValidationError(
                "evaluation result entry fields are malformed; "
                f"missing={missing}, extra={extra}"
            )
        prompt_id = item.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise EvaluationValidationError(
                "evaluation result prompt_id must be non-empty"
            )
        if prompt_id in prompt_ids:
            raise EvaluationValidationError(
                f"duplicate prompt ID in report: {prompt_id!r}"
            )
        prompt_ids.add(prompt_id)
        if prompt_id != expected_case.prompt_id or item.get("prompt_text") != expected_case.text:
            raise EvaluationValidationError(
                "evaluation result ordering or prompt identity does not match the suite"
            )
        if not isinstance(item.get("prompt_text"), str) or not isinstance(
            item.get("response"), str
        ):
            raise EvaluationValidationError(
                "evaluation result prompt and response must be text"
            )
        expected_prompt_digest = canonical_sha256(
            {"prompt_id": prompt_id, "prompt_text": item["prompt_text"]}
        )
        if item.get("prompt_digest") != expected_prompt_digest:
            raise EvaluationValidationError(
                "evaluation result prompt digest mismatch"
            )
        if not isinstance(item.get("stop_reason"), str) or not item["stop_reason"].strip():
            raise EvaluationValidationError(
                "evaluation result stop_reason must be non-empty"
            )
        token_count = item.get("generated_token_count")
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise EvaluationValidationError(
                "evaluation result token count is malformed"
            )
        rubric = _require_mapping(item.get("rubric"), "evaluation result rubric")
        if set(rubric) != {
            "version",
            "status",
            "passed",
            "matched_required_concepts",
            "missing_required_concepts",
            "forbidden_matches",
            "human_review_required",
        }:
            raise EvaluationValidationError(
                "evaluation result rubric fields are malformed"
            )
        if rubric.get("version") != RUBRIC_VERSION:
            raise EvaluationValidationError(
                f"unsupported scoring/rubric version: {rubric.get('version')!r}"
            )
        status = rubric.get("status")
        if status not in {"pass", "fail", "unscored", "human-review"}:
            raise EvaluationValidationError(f"invalid rubric status: {status!r}")
        if status in {"pass", "fail"} and not isinstance(
            rubric.get("passed"), bool
        ):
            raise EvaluationValidationError(
                "scored rubric result must contain boolean passed"
            )
        if status == "pass" and rubric.get("passed") is not True:
            raise EvaluationValidationError("passing rubric result must set passed=true")
        if status == "fail" and rubric.get("passed") is not False:
            raise EvaluationValidationError("failing rubric result must set passed=false")
        if status in {"unscored", "human-review"} and rubric.get("passed") is not None:
            raise EvaluationValidationError(
                "unscored rubric result must use null passed"
            )
        for field in (
            "matched_required_concepts",
            "missing_required_concepts",
            "forbidden_matches",
        ):
            if not isinstance(rubric.get(field), list) or not all(
                isinstance(value, str) for value in rubric[field]
            ):
                raise EvaluationValidationError(
                    f"evaluation rubric {field} must be a text list"
                )
        if not isinstance(rubric.get("human_review_required"), bool):
            raise EvaluationValidationError(
                "evaluation rubric human_review_required must be boolean"
            )
        if not isinstance(item.get("warnings"), list) or not all(
            isinstance(value, str) for value in item["warnings"]
        ):
            raise EvaluationValidationError(
                "evaluation result warnings must be a text list"
            )
        statuses.append(status)

    summary = _require_mapping(report.get("summary"), "evaluation report summary")
    if set(summary) != {
        "metric_label",
        "total_cases",
        "passed",
        "failed",
        "unscored",
        "held_out_generalization_measured",
    }:
        raise EvaluationValidationError("evaluation summary fields are malformed")
    expected = {
        "total_cases": len(results),
        "passed": statuses.count("pass"),
        "failed": statuses.count("fail"),
        "unscored": len(results) - statuses.count("pass") - statuses.count("fail"),
    }
    if any(summary.get(field) != value for field, value in expected.items()):
        raise EvaluationValidationError(
            "evaluation aggregate counts do not match results"
        )
    if not isinstance(summary.get("metric_label"), str) or not isinstance(
        summary.get("held_out_generalization_measured"), bool
    ):
        raise EvaluationValidationError(
            "evaluation summary metadata is malformed"
        )
    expected_label = _metric_label(registered_suite, contamination)
    expected_held_out_measured = (
        registered_suite.suite_id != ANCHOR_RETENTION_SUITE
        and contamination["status"] == "verified_clean"
    )
    if (
        summary["metric_label"] != expected_label
        or summary["held_out_generalization_measured"]
        is not expected_held_out_measured
    ):
        raise EvaluationValidationError(
            "evaluation summary wording or held-out classification is inconsistent"
        )
    if (
        registered_suite.suite_id == ANCHOR_RETENTION_SUITE
        and contamination["status"] != "contaminated"
    ):
        raise EvaluationValidationError(
            "historical Anchor reports must remain contaminated retention results"
        )
    if not isinstance(report.get("warnings"), list) or not all(
        isinstance(item, str) for item in report["warnings"]
    ):
        raise EvaluationValidationError(
            "evaluation report warnings must be a text list"
        )

    digest = report.get("digest")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise EvaluationValidationError(
            "evaluation report digest must be lowercase SHA-256"
        )
    payload = {key: value for key, value in report.items() if key != "digest"}
    if canonical_sha256(payload) != digest:
        raise EvaluationValidationError("evaluation report digest mismatch")


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_evaluation_report(value)
    return value


def write_evaluation_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    validate_evaluation_report(report)
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"evaluation report already exists: {destination.name}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_evaluation_report(report: Mapping[str, Any]) -> str:
    validate_evaluation_report(report)
    lines: list[str] = []
    for index, result in enumerate(report["results"], start=1):
        status = result["rubric"]["status"].upper()
        lines.extend(
            [
                f"[{index}] {result['prompt_id']} [{status}]",
                f"Prompt: {result['prompt_text']}",
                f"Answer: {result['response']}",
                f"Stop: {result['stop_reason']} "
                f"({result['generated_token_count']} tokens)",
                "",
            ]
        )
    summary = report["summary"]
    lines.append(
        f"{summary['metric_label']}: {summary['passed']}/{summary['total_cases']}."
    )
    lines.append(f"Held-out status: {report['contamination']['held_out_status']}.")
    if not summary["held_out_generalization_measured"]:
        lines.append("Held-out generalization: not yet measured.")
    for warning in report["warnings"]:
        lines.append(f"Warning: {warning}")
    lines.append(f"Report digest: {report['digest']}")
    return "\n".join(lines)


def format_chat_prompt(prompt: str, *, version: int = 1) -> str:
    if version != 1:
        raise EvaluationValidationError(
            f"unsupported prompt format version: {version}"
        )
    return f"<|user|>\n{prompt.strip()}\n<|assistant|>\n"


def _clean_response(text: str) -> str:
    cleaned = text.replace("<|assistant|>", "").replace("<|end|>", "")
    cleaned = cleaned.replace("<s>", "").replace("</s>", "")
    return cleaned.strip()


def torch_batch_generator(model: Any, tokenizer: Any) -> EvaluationBatchGenerator:
    """Adapt ByteSeed's generate API without changing generation semantics."""

    def generate(
        cases: tuple[EvaluationPrompt, ...],
        config: GenerationConfig,
    ) -> Sequence[GeneratedCaseOutput]:
        outputs: list[GeneratedCaseOutput] = []
        stop_ids = set(config.stop_token_ids) if config.stop_at_end else None
        eos_id = getattr(tokenizer, "eos_id", None)
        if stop_ids is not None and eos_id is not None:
            stop_ids.add(int(eos_id))
        device = torch.device(config.device)
        for case in cases:
            prompt_ids = tokenizer.encode(
                format_chat_prompt(
                    case.text,
                    version=config.prompt_format_version,
                ),
                add_bos=True,
                add_eos=False,
            )
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = model.generate(
                    idx,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    vocab_limit=getattr(tokenizer, "vocab_size", None),
                    stop_token_ids=stop_ids,
                )
            new_ids = generated[0, len(prompt_ids) :].tolist()
            if new_ids and stop_ids and new_ids[-1] in stop_ids:
                stop_reason = "stop_token"
            elif len(new_ids) >= config.max_new_tokens:
                stop_reason = "max_new_tokens"
            else:
                stop_reason = "model_complete"
            outputs.append(
                GeneratedCaseOutput(
                    response=_clean_response(tokenizer.decode(new_ids)),
                    generated_token_count=len(new_ids),
                    stop_reason=stop_reason,
                )
            )
        return outputs

    return generate


def logical_checkpoint_identity(
    checkpoint_path: str | Path,
    *,
    version: int | None,
    kind: str,
    legacy: bool,
    progress: int | None,
    data_manifest_digest: str | None = None,
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Construct path-safe checkpoint metadata for public reports."""

    logical_name = PurePath(checkpoint_path).name
    return {
        "logical_name": logical_name,
        "version": version,
        "kind": kind,
        "legacy": legacy,
        "progress": progress,
        "data_manifest_digest": data_manifest_digest,
        "artifact_sha256": artifact_sha256,
    }
