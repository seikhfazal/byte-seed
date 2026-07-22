"""Deterministic builder and provenance contracts for generalization-sft-v1."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import (
    NEAR_DUPLICATE_THRESHOLD,
    NEAR_DUPLICATE_VERSION,
    NearDuplicateFinding,
    create_document,
    detect_evaluation_contamination,
    detect_near_duplicate_texts,
    normalize_overlap_text,
    plan_document_dataset,
    read_sft_documents,
)
from .eval_prompts import (
    EVALUATION_PROMPT_REGISTRY_VERSION,
    registered_evaluation_prompts,
    registered_evaluation_suites,
    serialize_evaluation_suite,
)
from .finetune_chat import format_chat
from .generalization_sft_source import (
    DATASET_NAME,
    DATASET_VERSION,
    LESSONS,
    PROMPT_FORMS,
    REQUIRED_FAMILIES,
    SOURCE_VERSION,
    lesson_cluster_id,
    render_prompt,
    render_response,
)
from .provenance import HASH_ALGORITHM, canonical_json_bytes, canonical_sha256, sha256_file


BUILDER_VERSION = 1
SFT_RECORD_SCHEMA_VERSION = 1
SFT_MANIFEST_VERSION = 1
SFT_QUALITY_REPORT_VERSION = 1
SFT_MANIFEST_KIND = "sft_data_manifest"
SFT_QUALITY_REPORT_KIND = "sft_data_quality"
DEFAULT_GROUP_SPLIT_SEED = 20260722
DEFAULT_GROUP_VALIDATION_RATIO = 0.1
SPLIT_PREVIEW_VERSION = 1
INTERNAL_NEAR_REVIEW_VERSION = 1
APPROVED_CROSS_TOPIC_NEAR_PAIRS = frozenset(
    {
        (
            "gen-v1.queue-dequeue.example_request",
            "gen-v1.stack-pop.example_request",
        ),
        (
            "gen-v1.queue-dequeue.example_request",
            "gen-v1.stack-push.example_request",
        ),
        (
            "gen-v1.queue-empty.direct_definition",
            "gen-v1.stack-empty.direct_definition",
        ),
        (
            "gen-v1.queue-empty.example_request",
            "gen-v1.stack-empty.example_request",
        ),
        (
            "gen-v1.queue-enqueue.example_request",
            "gen-v1.stack-pop.example_request",
        ),
        (
            "gen-v1.queue-enqueue.example_request",
            "gen-v1.stack-push.example_request",
        ),
        (
            "gen-v1.queue-front.example_request",
            "gen-v1.stack-peek.example_request",
        ),
    }
)
RECORD_KEYS = frozenset(
    {
        "id", "dataset", "source", "category", "prompt_form", "group_id",
        "user", "assistant", "text",
    }
)
CONTROL_TOKENS = ("<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
PLACEHOLDERS = ("todo", "tbd", "lorem ipsum")


class GeneralizationSFTValidationError(ValueError):
    """Raised when the versioned SFT source or artifacts fail closed."""


def build_generalization_records() -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    for lesson in LESSONS:
        for prompt_form in PROMPT_FORMS:
            user = render_prompt(lesson, prompt_form)
            assistant = render_response(lesson, prompt_form)
            records.append(
                {
                    "id": f"gen-v1.{lesson.lesson_id}.{prompt_form}",
                    "dataset": DATASET_VERSION,
                    "source": f"{DATASET_VERSION}/{lesson.lesson_id}",
                    "category": lesson.family,
                    "prompt_form": prompt_form,
                    "group_id": (
                        f"{DATASET_VERSION}:{lesson.family}:{lesson_cluster_id(lesson)}"
                    ),
                    "user": user,
                    "assistant": assistant,
                    "text": format_chat(user, assistant),
                }
            )
    result = tuple(records)
    validate_generalization_records(result)
    return result


def validate_generalization_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise GeneralizationSFTValidationError("SFT records must be a non-empty sequence")
    ids: set[str] = set()
    prompts: set[tuple[str, ...]] = set()
    conversations: set[str] = set()
    families: Counter[str] = Counter()
    forms: Counter[str] = Counter()
    group_families: dict[str, str] = {}
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, Mapping):
            raise GeneralizationSFTValidationError(f"record {index} must be an object")
        if set(raw) != RECORD_KEYS:
            missing = sorted(RECORD_KEYS - set(raw))
            extra = sorted(set(raw) - RECORD_KEYS)
            raise GeneralizationSFTValidationError(
                f"record {index} schema mismatch; missing={missing}, extra={extra}"
            )
        for field in RECORD_KEYS:
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise GeneralizationSFTValidationError(
                    f"record {index} field {field!r} must be non-empty text"
                )
        record_id = raw["id"]
        if record_id in ids:
            raise GeneralizationSFTValidationError(f"duplicate record ID: {record_id}")
        ids.add(record_id)
        if raw["dataset"] != DATASET_VERSION:
            raise GeneralizationSFTValidationError(f"record {record_id} has wrong dataset version")
        family = raw["category"]
        prompt_form = raw["prompt_form"]
        if family not in REQUIRED_FAMILIES:
            raise GeneralizationSFTValidationError(f"record {record_id} has invalid concept family")
        if prompt_form not in PROMPT_FORMS:
            raise GeneralizationSFTValidationError(f"record {record_id} has invalid prompt form")
        if raw["text"] != format_chat(raw["user"], raw["assistant"]):
            raise GeneralizationSFTValidationError(f"record {record_id} has malformed chat text")
        combined = f"{raw['user']}\n{raw['assistant']}".casefold()
        if any(token.casefold() in combined for token in CONTROL_TOKENS):
            raise GeneralizationSFTValidationError(f"record {record_id} leaks a chat control token")
        if any(marker in combined for marker in PLACEHOLDERS):
            raise GeneralizationSFTValidationError(f"record {record_id} contains placeholder text")
        if "python chat.py.py" in combined or "pythonchat.py" in combined:
            raise GeneralizationSFTValidationError(f"record {record_id} has a malformed chat command")
        if "torch.cuda.isavailable" in combined or "torch.cuda.available" in combined:
            raise GeneralizationSFTValidationError(f"record {record_id} has a malformed CUDA API")
        answer = raw["assistant"].casefold()
        if family == "overfitting" and any(
            phrase in answer
            for phrase in ("this is underfitting", "indicates underfitting", "called underfitting")
        ):
            raise GeneralizationSFTValidationError(
                f"record {record_id} contradicts the overfitting family"
            )
        if family == "underfitting" and any(
            phrase in answer
            for phrase in ("this is overfitting", "indicates overfitting", "called overfitting")
        ):
            raise GeneralizationSFTValidationError(
                f"record {record_id} contradicts the underfitting family"
            )
        if len(raw["text"]) > 1800:
            raise GeneralizationSFTValidationError(f"record {record_id} is excessively long")
        prompt_identity = normalize_overlap_text(raw["user"])
        if prompt_identity in prompts:
            raise GeneralizationSFTValidationError(
                f"duplicate normalized user prompt: {record_id}"
            )
        prompts.add(prompt_identity)
        conversation = canonical_sha256(
            {"user": raw["user"], "assistant": raw["assistant"]}
        )
        if conversation in conversations:
            raise GeneralizationSFTValidationError(
                f"duplicate complete conversation: {record_id}"
            )
        conversations.add(conversation)
        previous_family = group_families.setdefault(raw["group_id"], family)
        if previous_family != family:
            raise GeneralizationSFTValidationError(
                f"group {raw['group_id']} crosses concept families"
            )
        families[family] += 1
        forms[prompt_form] += 1

    if set(families) != set(REQUIRED_FAMILIES):
        raise GeneralizationSFTValidationError("not every required concept family is present")
    if not 750 <= len(records) <= 1100:
        raise GeneralizationSFTValidationError("record count must be between 750 and 1100")
    for family in REQUIRED_FAMILIES:
        if families[family] < 50:
            raise GeneralizationSFTValidationError(f"concept family {family} has fewer than 50 records")
        if families[family] / len(records) > 0.15:
            raise GeneralizationSFTValidationError(f"concept family {family} exceeds 15 percent")


def assign_group_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_GROUP_SPLIT_SEED,
    validation_ratio: float = DEFAULT_GROUP_VALIDATION_RATIO,
) -> dict[str, str]:
    validate_generalization_records(records)
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation ratio must be between zero and one")
    group_families: dict[str, str] = {}
    for record in records:
        group_id = str(record["group_id"])
        family = str(record["category"])
        previous = group_families.setdefault(group_id, family)
        if previous != family:
            raise GeneralizationSFTValidationError(
                f"group {group_id!r} crosses concept families"
            )
    groups = sorted(group_families)
    assignments: dict[str, str] = {}
    scores: dict[str, float] = {}
    for group_id in groups:
        payload = f"{SPLIT_PREVIEW_VERSION}\0{seed}\0{group_id}".encode("utf-8")
        score = int.from_bytes(hashlib.sha256(payload).digest(), "big") / float(1 << 256)
        scores[group_id] = score
        assignments[group_id] = "validation" if score < validation_ratio else "train"
    for family in REQUIRED_FAMILIES:
        family_groups = sorted(
            group_id for group_id, value in group_families.items() if value == family
        )
        if len(family_groups) < 2:
            raise GeneralizationSFTValidationError(
                f"concept family {family} requires multiple split groups"
            )
        if not any(assignments[group_id] == "validation" for group_id in family_groups):
            selected = min(
                family_groups,
                key=lambda value: (scores[value], value),
            )
            assignments[selected] = "validation"
        if not any(assignments[group_id] == "train" for group_id in family_groups):
            selected = max(
                family_groups,
                key=lambda value: (scores[value], value),
            )
            assignments[selected] = "train"
    return dict(sorted(assignments.items()))


def build_split_readiness(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_GROUP_SPLIT_SEED,
    validation_ratio: float = DEFAULT_GROUP_VALIDATION_RATIO,
) -> dict[str, Any]:
    """Apply canonical group assignments through the existing document-aware split."""

    validate_generalization_records(records)
    group_assignments = assign_group_splits(
        records,
        seed=seed,
        validation_ratio=validation_ratio,
    )
    documents = [
        create_document(
            text=str(record["text"]),
            source=str(record["source"]),
            document_id=str(record["id"]),
            fields={
                "user": str(record["user"]),
                "assistant": str(record["assistant"]),
            },
            explicit_split=group_assignments[str(record["group_id"])],
        )
        for record in records
    ]
    plan = plan_document_dataset(
        documents,
        seed=seed,
        validation_ratio=validation_ratio,
        prompts=(),
    )
    record_by_id = {str(record["id"]): record for record in records}
    train_ids = {document.document_id for document in plan.train_documents}
    validation_ids = {document.document_id for document in plan.validation_documents}
    if not train_ids or not validation_ids:
        raise GeneralizationSFTValidationError("canonical SFT split produced an empty partition")
    if train_ids & validation_ids:
        raise GeneralizationSFTValidationError("record IDs leak across the canonical SFT split")

    group_records: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        group_records.setdefault(str(record["group_id"]), []).append(record)
    group_entries: list[dict[str, Any]] = []
    train_groups: set[str] = set()
    validation_groups: set[str] = set()
    for group_id, members in sorted(group_records.items()):
        member_ids = {str(member["id"]) for member in members}
        in_train = bool(member_ids & train_ids)
        in_validation = bool(member_ids & validation_ids)
        if in_train == in_validation:
            raise GeneralizationSFTValidationError(
                f"group {group_id!r} is empty or leaks across train and validation"
            )
        split = "train" if in_train else "validation"
        if split != group_assignments[group_id]:
            raise GeneralizationSFTValidationError(
                f"document-aware split disagrees with group assignment for {group_id!r}"
            )
        (train_groups if split == "train" else validation_groups).add(group_id)
        group_entries.append(
            {
                "group_id": group_id,
                "family": str(members[0]["category"]),
                "record_count": len(members),
                "split": split,
            }
        )
    leakage = sorted(train_groups & validation_groups)
    if leakage:
        raise GeneralizationSFTValidationError(
            "group leakage across train and validation: " + ", ".join(leakage)
        )

    family_counts = {
        "train": dict(
            sorted(Counter(str(record_by_id[item]["category"]) for item in train_ids).items())
        ),
        "validation": dict(
            sorted(
                Counter(str(record_by_id[item]["category"]) for item in validation_ids).items()
            )
        ),
    }
    for split in ("train", "validation"):
        if set(family_counts[split]) != set(REQUIRED_FAMILIES):
            raise GeneralizationSFTValidationError(
                f"canonical {split} split does not cover every concept family"
            )
    group_sizes = [entry["record_count"] for entry in group_entries]
    group_counts = Counter(entry["family"] for entry in group_entries)
    preview: dict[str, Any] = {
        "version": SPLIT_PREVIEW_VERSION,
        "strategy": "group-id-sha256-with-family-coverage",
        "document_split_strategy": "canonical-group-sha256",
        "seed": seed,
        "validation_ratio": float(validation_ratio),
        "record_counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
        },
        "family_counts": family_counts,
        "group_counts": dict(sorted(group_counts.items())),
        "group_size_distribution": {
            "minimum": min(group_sizes),
            "median": float(statistics.median(group_sizes)),
            "mean": round(statistics.mean(group_sizes), 3),
            "maximum": max(group_sizes),
        },
        "group_leakage": leakage,
        "groups": group_entries,
    }
    preview["digest"] = canonical_sha256(preview)
    return preview


def review_internal_near_findings(
    records: Sequence[Mapping[str, Any]],
    findings: Sequence[NearDuplicateFinding],
) -> tuple[dict[str, Any], ...]:
    """Classify every finding; sharing a group alone never grants an exception."""

    by_id = {str(record["id"]): record for record in records}
    reviewed: list[dict[str, Any]] = []
    approved_merged_clusters = {
        f"{DATASET_VERSION}:stack_fundamentals:stack-push-pop",
        f"{DATASET_VERSION}:queue_fundamentals:queue-operations",
    }
    for finding in findings:
        try:
            left = by_id[finding.left_id]
            right = by_id[finding.right_id]
        except KeyError as exc:
            raise GeneralizationSFTValidationError(
                f"near-duplicate review references unknown record {exc.args[0]!r}"
            ) from exc
        left_family = str(left["category"])
        right_family = str(right["category"])
        left_group = str(left["group_id"])
        right_group = str(right["group_id"])
        same_lesson = str(left["source"]) == str(right["source"])
        approved_cluster = left_group == right_group and left_group in approved_merged_clusters
        if same_lesson or approved_cluster:
            classification = "expected same-cluster variant"
            rationale = (
                "different prompt forms for one authored lesson"
                if same_lesson
                else "complementary operations reviewed as one semantic cluster"
            )
        elif tuple(sorted((finding.left_id, finding.right_id))) in (
            APPROVED_CROSS_TOPIC_NEAR_PAIRS
        ):
            classification = "legitimate cross-topic wording"
            rationale = (
                "parallel stack/queue terminology is intentional and the responses "
                "state different structures or operations"
            )
        else:
            classification = "rewrite required"
            rationale = "near wording is not covered by an explicitly reviewed semantic relationship"
        reviewed.append(
            {
                "left_id": finding.left_id,
                "right_id": finding.right_id,
                "left_family": left_family,
                "right_family": right_family,
                "left_group_id": left_group,
                "right_group_id": right_group,
                "similarity": finding.similarity,
                "classification": classification,
                "rationale": rationale,
            }
        )
    return tuple(reviewed)


def jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    validate_generalization_records(records)
    lines = [
        json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_generalization_artifacts(
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    quality_report_path: str | Path,
    source_path: str | Path,
    curated_core_path: str | Path,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    records = build_generalization_records()
    encoded = jsonl_bytes(records)
    output_digest = hashlib.sha256(encoded).hexdigest()
    curated_rows = _read_curated_core(curated_core_path)
    quality = _build_quality_report(records, curated_rows, output_digest)
    manifest = _build_manifest(
        records,
        encoded,
        quality,
        output_path=output_path,
        quality_report_path=quality_report_path,
        source_path=source_path,
        curated_core_path=curated_core_path,
        curated_rows=curated_rows,
    )
    validate_sft_quality_report(quality)
    validate_sft_manifest(manifest, quality_report=quality)
    return encoded, manifest, quality


def write_generalization_artifacts(
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    quality_report_path: str | Path,
    source_path: str | Path,
    curated_core_path: str | Path,
    overwrite: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    destinations = tuple(Path(value) for value in (output_path, manifest_path, quality_report_path))
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite existing artifact(s): " + ", ".join(map(str, existing)))
    encoded, manifest, quality = build_generalization_artifacts(
        output_path=output_path,
        manifest_path=manifest_path,
        quality_report_path=quality_report_path,
        source_path=source_path,
        curated_core_path=curated_core_path,
    )
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(encoded)
    Path(manifest_path).write_bytes(canonical_json_bytes(manifest) + b"\n")
    Path(quality_report_path).write_bytes(canonical_json_bytes(quality) + b"\n")
    return manifest, quality


def validate_sft_artifact_files(
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    quality_report_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = Path(output_path)
    manifest_file = Path(manifest_path)
    quality_file = Path(quality_report_path)
    if quality_file.is_file() and not manifest_file.is_file():
        raise GeneralizationSFTValidationError(
            "orphaned SFT data-quality report found; the versioned SFT manifest is missing"
        )
    if manifest_file.is_file() and not quality_file.is_file():
        raise GeneralizationSFTValidationError(
            "SFT manifest exists but its linked data-quality report is missing"
        )
    if not manifest_file.is_file() or not quality_file.is_file():
        raise GeneralizationSFTValidationError("SFT manifest and data-quality report are required")
    if not output.is_file():
        raise GeneralizationSFTValidationError("SFT JSONL output referenced by the manifest is missing")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        quality = json.loads(quality_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneralizationSFTValidationError("could not load SFT provenance artifacts") from exc
    validate_sft_manifest(manifest, quality_report=quality)
    if sha256_file(output) != manifest["output"]["sha256"]:
        raise GeneralizationSFTValidationError("SFT JSONL digest does not match its manifest")
    return manifest, quality


def validate_sft_quality_report(report: Mapping[str, Any]) -> None:
    required = {
        "version", "kind", "dataset", "builder", "output_sha256", "counts",
        "family_counts", "prompt_form_counts", "response_lengths", "deduplication",
        "split_readiness", "cross_dataset", "evaluation_audit", "policy", "digest",
    }
    _require_exact_keys(report, required, "SFT quality report")
    if report["version"] != SFT_QUALITY_REPORT_VERSION or report["kind"] != SFT_QUALITY_REPORT_KIND:
        raise GeneralizationSFTValidationError("unsupported SFT quality-report version or kind")
    if report["dataset"] != {"name": DATASET_NAME, "version": DATASET_VERSION}:
        raise GeneralizationSFTValidationError("SFT quality-report dataset identity is invalid")
    _validate_digest(report["output_sha256"], "SFT JSONL digest")
    audit = report["evaluation_audit"]
    if not isinstance(audit, Mapping) or audit.get("registry_version") != EVALUATION_PROMPT_REGISTRY_VERSION:
        raise GeneralizationSFTValidationError("evaluation suite coverage is incomplete")
    expected = [
        (suite.suite_id, suite.version, [case.prompt_id for case in suite.cases])
        for suite in registered_evaluation_suites()
    ]
    actual = [
        (entry.get("suite_id"), entry.get("suite_version"), entry.get("prompt_ids"))
        for entry in audit.get("suites", [])
    ]
    if actual != expected:
        raise GeneralizationSFTValidationError("evaluation suite coverage is incomplete or reordered")
    for entry, suite in zip(audit["suites"], registered_evaluation_suites(), strict=True):
        if entry.get("suite_digest") != canonical_sha256(serialize_evaluation_suite(suite)):
            raise GeneralizationSFTValidationError("evaluation suite digest mismatch")
    if audit.get("exact_findings") or audit.get("near_findings"):
        raise GeneralizationSFTValidationError("evaluation contamination is present")
    split = report["split_readiness"]
    if not isinstance(split, Mapping):
        raise GeneralizationSFTValidationError("SFT split-readiness preview must be a mapping")
    split_digest = _validate_digest(split.get("digest"), "SFT split-readiness digest")
    if split_digest != canonical_sha256(
        {key: value for key, value in split.items() if key != "digest"}
    ):
        raise GeneralizationSFTValidationError("SFT split-readiness digest mismatch")
    if split.get("group_leakage") != []:
        raise GeneralizationSFTValidationError("SFT split-readiness contains group leakage")
    for partition in ("train", "validation"):
        if set(split.get("family_counts", {}).get(partition, {})) != set(REQUIRED_FAMILIES):
            raise GeneralizationSFTValidationError(
                f"SFT {partition} split does not cover every concept family"
            )
    deduplication = report["deduplication"]
    review = deduplication.get("internal_near_review", [])
    if len(review) != deduplication.get("internal_near_count"):
        raise GeneralizationSFTValidationError("internal near-review coverage is incomplete")
    if any(item.get("classification") == "rewrite required" for item in review):
        raise GeneralizationSFTValidationError("internal near review still requires rewrites")
    stored = _validate_digest(report["digest"], "SFT quality-report digest")
    if stored != canonical_sha256({key: value for key, value in report.items() if key != "digest"}):
        raise GeneralizationSFTValidationError("SFT quality-report digest mismatch")


def validate_sft_manifest(
    manifest: Mapping[str, Any],
    *,
    quality_report: Mapping[str, Any] | None,
) -> None:
    required = {
        "version", "kind", "dataset", "builder", "record_schema", "source_files",
        "build_configuration", "output", "grouping", "intended_training_components", "evaluation_audit",
        "quality_report", "digest",
    }
    _require_exact_keys(manifest, required, "SFT manifest")
    if manifest["version"] != SFT_MANIFEST_VERSION or manifest["kind"] != SFT_MANIFEST_KIND:
        raise GeneralizationSFTValidationError("unsupported SFT manifest version or kind")
    stored = _validate_digest(manifest["digest"], "SFT manifest digest")
    if stored != canonical_sha256({key: value for key, value in manifest.items() if key != "digest"}):
        raise GeneralizationSFTValidationError("SFT manifest digest mismatch")
    if quality_report is None:
        raise GeneralizationSFTValidationError(
            "orphaned SFT manifest: the linked data-quality report is missing"
        )
    validate_sft_quality_report(quality_report)
    if manifest["quality_report"].get("digest") != quality_report["digest"]:
        raise GeneralizationSFTValidationError("manifest-to-quality-report digest mismatch")


def _build_quality_report(
    records: Sequence[Mapping[str, Any]],
    curated_rows: Sequence[Mapping[str, str]],
    output_digest: str,
) -> dict[str, Any]:
    record_texts = [(str(row["id"]), str(row["user"])) for row in records]
    internal_near = detect_near_duplicate_texts(record_texts)
    internal_review = review_internal_near_findings(records, internal_near)
    review_counts = Counter(item["classification"] for item in internal_review)
    rewrite_required = [
        item for item in internal_review if item["classification"] == "rewrite required"
    ]
    record_groups = {str(row["id"]): str(row["group_id"]) for row in records}
    cross_group_near = [
        finding
        for finding in internal_near
        if record_groups[finding.left_id] != record_groups[finding.right_id]
    ]
    core_texts = [
        (f"curated-core-{index:04d}", row["user"])
        for index, row in enumerate(curated_rows, start=1)
    ]
    cross_near = detect_near_duplicate_texts(record_texts, core_texts)
    prompt_ids = {normalize_overlap_text(str(row["user"])) for row in records}
    core_prompt_ids = {normalize_overlap_text(row["user"]) for row in curated_rows}
    conversations = {
        canonical_sha256({"user": row["user"], "assistant": row["assistant"]})
        for row in records
    }
    core_conversations = {
        canonical_sha256({"user": row["user"], "assistant": row["assistant"]})
        for row in curated_rows
    }
    documents = _records_as_documents(records)
    exact_findings = [finding.as_dict() for finding in detect_evaluation_contamination(documents)]
    suite_prompts = [
        (prompt.prompt_id, prompt.text) for prompt in registered_evaluation_prompts()
    ]
    near_findings = [
        finding.as_dict()
        for finding in detect_near_duplicate_texts(record_texts, suite_prompts)
    ]
    family_counts = Counter(str(row["category"]) for row in records)
    form_counts = Counter(str(row["prompt_form"]) for row in records)
    lengths = [len(str(row["assistant"]).split()) for row in records]
    response_lengths = {
        "unit": "words",
        "minimum": min(lengths),
        "maximum": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 3),
        "buckets": {
            "1-12": sum(value <= 12 for value in lengths),
            "13-24": sum(13 <= value <= 24 for value in lengths),
            "25-40": sum(25 <= value <= 40 for value in lengths),
            "41+": sum(value >= 41 for value in lengths),
        },
    }
    split_readiness = build_split_readiness(records)
    report: dict[str, Any] = {
        "version": SFT_QUALITY_REPORT_VERSION,
        "kind": SFT_QUALITY_REPORT_KIND,
        "dataset": {"name": DATASET_NAME, "version": DATASET_VERSION},
        "builder": {"version": BUILDER_VERSION, "source_version": SOURCE_VERSION},
        "output_sha256": output_digest,
        "counts": {
            "records": len(records),
            "normalized_unique_prompts": len(prompt_ids),
            "exact_duplicate_prompts": len(records) - len(prompt_ids),
            "exact_duplicate_conversations": len(records) - len(conversations),
        },
        "family_counts": dict(sorted(family_counts.items())),
        "prompt_form_counts": dict(sorted(form_counts.items())),
        "response_lengths": response_lengths,
        "split_readiness": split_readiness,
        "deduplication": {
            "near_duplicate_version": NEAR_DUPLICATE_VERSION,
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "review_version": INTERNAL_NEAR_REVIEW_VERSION,
            "internal_near_count": len(internal_near),
            "within_group_near_count": len(internal_near) - len(cross_group_near),
            "cross_group_near_count": len(cross_group_near),
            "internal_near_findings": [item.as_dict() for item in internal_near],
            "review_summary": {
                classification: review_counts.get(classification, 0)
                for classification in (
                    "expected same-cluster variant",
                    "legitimate cross-topic wording",
                    "rewrite required",
                )
            },
            "internal_near_review": list(internal_review),
        },
        "cross_dataset": {
            "component": "curated-personal-assistant-core",
            "record_count": len(curated_rows),
            "exact_prompt_overlap_count": len(prompt_ids & core_prompt_ids),
            "exact_conversation_overlap_count": len(conversations & core_conversations),
            "near_prompt_overlap_count": len(cross_near),
            "near_prompt_findings": [item.as_dict() for item in cross_near],
        },
        "evaluation_audit": {
            "registry_version": EVALUATION_PROMPT_REGISTRY_VERSION,
            "suites": [
                {
                    "suite_id": suite.suite_id,
                    "suite_version": suite.version,
                    "suite_digest": canonical_sha256(serialize_evaluation_suite(suite)),
                    "prompt_ids": [case.prompt_id for case in suite.cases],
                }
                for suite in registered_evaluation_suites()
            ],
            "exact_findings": exact_findings,
            "near_findings": near_findings,
        },
        "policy": {
            "evaluation_contamination_allowed": False,
            "concept_overlap_intentional": True,
            "exact_or_near_evaluation_wording_forbidden": True,
            "holdout_excluded_from_training": True,
        },
    }
    report["digest"] = canonical_sha256(report)
    if exact_findings or near_findings:
        raise GeneralizationSFTValidationError(
            "evaluation contamination detected; exact and near wording overlap are forbidden"
        )
    if rewrite_required:
        pairs = ", ".join(
            f"{item['left_id']} / {item['right_id']}" for item in rewrite_required[:5]
        )
        raise GeneralizationSFTValidationError(
            "internal near-duplicate review requires source rewrites: " + pairs
        )
    return report


def _build_manifest(
    records: Sequence[Mapping[str, Any]],
    encoded: bytes,
    quality: Mapping[str, Any],
    *,
    output_path: str | Path,
    quality_report_path: str | Path,
    source_path: str | Path,
    curated_core_path: str | Path,
    curated_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    output_digest = hashlib.sha256(encoded).hexdigest()
    manifest: dict[str, Any] = {
        "version": SFT_MANIFEST_VERSION,
        "kind": SFT_MANIFEST_KIND,
        "dataset": {"name": DATASET_NAME, "version": DATASET_VERSION},
        "builder": {"version": BUILDER_VERSION, "source_version": SOURCE_VERSION},
        "record_schema": {
            "version": SFT_RECORD_SCHEMA_VERSION,
            "required_fields": sorted(RECORD_KEYS),
            "encoding": "utf-8",
            "newline": "lf",
        },
        "source_files": [
            {
                "logical_name": "src/byteseed/generalization_sft_source.py",
                "sha256": sha256_file(source_path),
            }
        ],
        "build_configuration": {
            "randomness": "none",
            "record_order": "lesson-source-order-then-prompt-form-order",
            "prompt_forms": list(PROMPT_FORMS),
            "near_duplicate_version": NEAR_DUPLICATE_VERSION,
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "split_preview_version": SPLIT_PREVIEW_VERSION,
        },
        "output": {
            "logical_name": Path(output_path).name,
            "sha256": output_digest,
            "size_bytes": len(encoded),
            "record_count": len(records),
        },
        "grouping": {
            "strategy": "source-template-semantic-cluster",
            "group_field": "group_id",
            "group_count": len({row["group_id"] for row in records}),
            "group_counts": quality["split_readiness"]["group_counts"],
            "group_size_distribution": quality["split_readiness"][
                "group_size_distribution"
            ],
            "split_strategy": quality["split_readiness"]["strategy"],
            "split_seed": DEFAULT_GROUP_SPLIT_SEED,
            "split_preview_digest": quality["split_readiness"]["digest"],
        },
        "intended_training_components": [
            {
                "name": "curated-personal-assistant-core",
                "logical_name": "data/raw/assistant_sft/curated_personal_assistant_core.jsonl",
                "sha256": sha256_file(curated_core_path),
                "record_count": len(curated_rows),
            },
            {
                "name": DATASET_VERSION,
                "logical_name": Path(output_path).name,
                "sha256": output_digest,
                "record_count": len(records),
            },
        ],
        "evaluation_audit": quality["evaluation_audit"],
        "quality_report": {
            "version": SFT_QUALITY_REPORT_VERSION,
            "logical_name": Path(quality_report_path).name,
            "digest": quality["digest"],
        },
    }
    manifest["digest"] = canonical_sha256(manifest)
    return manifest


def _read_curated_core(path: str | Path) -> tuple[dict[str, str], ...]:
    source = Path(path)
    rows: list[dict[str, str]] = []
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise GeneralizationSFTValidationError(f"could not read curated core: {source}") from exc
    for number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GeneralizationSFTValidationError(f"invalid curated core JSON at line {number}") from exc
        if not isinstance(raw, Mapping):
            raise GeneralizationSFTValidationError(f"curated core line {number} must be an object")
        user, assistant = raw.get("user"), raw.get("assistant")
        if not isinstance(user, str) or not user.strip() or not isinstance(assistant, str) or not assistant.strip():
            raise GeneralizationSFTValidationError(
                f"curated core line {number} requires user and assistant text"
            )
        rows.append({"user": user, "assistant": assistant})
    if not rows:
        raise GeneralizationSFTValidationError("curated core is empty")
    return tuple(rows)


def _records_as_documents(records: Sequence[Mapping[str, Any]]):
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "generalization.jsonl"
        path.write_bytes(jsonl_bytes(records))
        return read_sft_documents(path)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise GeneralizationSFTValidationError(f"{label} fields are malformed")


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise GeneralizationSFTValidationError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GeneralizationSFTValidationError(f"{label} must be hexadecimal") from exc
    return value
