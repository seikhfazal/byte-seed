from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import warnings
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .eval_prompts import (
    EVALUATION_PROMPT_REGISTRY_VERSION,
    EvaluationPrompt,
    registered_evaluation_prompts,
    registered_evaluation_suites,
)
from .provenance import (
    HASH_ALGORITHM,
    ProvenanceValidationError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_logical_name,
)


DOCUMENT_FORMAT_VERSION = 1
NORMALIZATION_VERSION = 1
DEDUPLICATION_VERSION = 1
SPLIT_STRATEGY_VERSION = 1
DATA_QUALITY_REPORT_VERSION = 1
SPLIT_STRATEGY = "canonical-group-sha256"
DEDUPLICATION_STRATEGY = "canonical-sha256-representative"
NEAR_DUPLICATE_VERSION = 1
NEAR_DUPLICATE_THRESHOLD = 0.82

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_ORDINARY_WHITESPACE_RE = re.compile(r"[ \t]+")
_OVERLAP_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*")


class DataQualityError(RuntimeError):
    """Base error for document preparation and auditing failures."""


class DocumentValidationError(DataQualityError):
    """Raised when a source record cannot be represented safely."""


class DataLeakageError(DataQualityError):
    """Raised when document identity crosses train/validation boundaries."""


class EvaluationContaminationError(DataQualityError):
    """Raised when registered evaluation input occurs in new training material."""

    def __init__(self, findings: Sequence["ContaminationFinding"]):
        self.findings = tuple(findings)
        prompt_ids = sorted({finding.prompt_id for finding in findings})
        super().__init__(
            "Evaluation contamination detected for registered prompt(s): "
            + ", ".join(prompt_ids)
            + ". Re-run only for historical reproduction with "
            "--allow-eval-contamination; contaminated results are retention checks, "
            "not held-out generalization."
        )


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    source: str
    fields: tuple[tuple[str, str], ...]
    explicit_split: str | None
    raw_fingerprint: str
    canonical_fingerprint: str

    def field_map(self) -> dict[str, str]:
        return dict(self.fields)


@dataclass(frozen=True)
class DuplicateGroup:
    canonical_fingerprint: str
    representative: Document
    members: tuple[Document, ...]
    removed: tuple[Document, ...]


@dataclass(frozen=True)
class DuplicateAnalysis:
    groups: tuple[DuplicateGroup, ...]
    representatives: tuple[Document, ...]
    raw_duplicate_count: int
    canonical_duplicate_count: int

    @property
    def removed_count(self) -> int:
        return sum(len(group.removed) for group in self.groups)


@dataclass(frozen=True)
class ContaminationFinding:
    prompt_id: str
    suite: str
    dataset_split: str
    document_id: str
    source: str
    match_type: str
    document_fingerprint: str

    def as_dict(self) -> dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "suite": self.suite,
            "dataset_split": self.dataset_split,
            "document_id": self.document_id,
            "source": self.source,
            "match_type": self.match_type,
            "document_fingerprint": self.document_fingerprint,
        }


@dataclass(frozen=True)
class NearDuplicateFinding:
    left_id: str
    right_id: str
    similarity: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class DocumentPlan:
    documents: tuple[Document, ...]
    duplicates: DuplicateAnalysis
    train_documents: tuple[Document, ...]
    validation_documents: tuple[Document, ...]
    findings: tuple[ContaminationFinding, ...]
    audited_prompts: tuple[EvaluationPrompt, ...]
    assignments: Mapping[str, str]
    split_seed: int
    validation_ratio: float
    allow_eval_contamination: bool


def normalize_document_text(text: str) -> str:
    """Conservatively normalize prose while preserving indented/fenced code."""

    if not isinstance(text, str):
        raise DocumentValidationError("Document text must be a string.")
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    result: list[str] = []
    in_fence = False
    for line in lines:
        fence = _FENCE_RE.match(line)
        if in_fence or line.startswith(("    ", "\t")):
            cleaned = line.rstrip()
        else:
            cleaned = _ORDINARY_WHITESPACE_RE.sub(" ", line).strip()
        result.append(cleaned)
        if fence:
            in_fence = not in_fence
    return "\n".join(result)


def normalize_overlap_text(text: str) -> tuple[str, ...]:
    """Normalize prose for conservative near-wording comparisons.

    The canonical document normalizer remains unchanged. This secondary form
    deliberately ignores case and punctuation so cosmetic evaluation-prompt
    rewrites cannot evade a contamination audit.
    """

    normalized = unicodedata.normalize("NFKC", normalize_document_text(text)).casefold()
    tokens = _OVERLAP_TOKEN_RE.findall(normalized)
    return tuple(_singular_overlap_token(token) for token in tokens)


def near_duplicate_similarity(first: str, second: str) -> float:
    """Return a deterministic token similarity in the inclusive range 0..1."""

    left = normalize_overlap_text(first)
    right = normalize_overlap_text(second)
    return _near_duplicate_similarity_tokens(left, right)


def _near_duplicate_similarity_tokens(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    sequence_ratio = SequenceMatcher(None, left, right, autojunk=False).ratio()
    left_set = set(left)
    right_set = set(right)
    dice = (2.0 * len(left_set & right_set)) / (len(left_set) + len(right_set))
    left_counts = Counter(left)
    right_counts = Counter(right)
    bag_dice = (
        2.0 * sum((left_counts & right_counts).values()) / (len(left) + len(right))
    )
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    containment = 0.0
    if len(shorter) >= 2:
        window = len(shorter)
        if any(
            tuple(longer[index : index + window]) == shorter
            for index in range(len(longer) - window + 1)
        ):
            containment = 1.0
        elif len(set(shorter) & set(longer)) / len(set(shorter)) >= 0.9:
            containment = min(1.0, len(shorter) / max(1, len(longer)) + 0.35)
    return round(max(sequence_ratio, dice, bag_dice, containment), 6)


def detect_near_duplicate_texts(
    left: Iterable[tuple[str, str]],
    right: Iterable[tuple[str, str]] | None = None,
    *,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> tuple[NearDuplicateFinding, ...]:
    """Find near-identical wording within one collection or across two collections."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be in the range (0, 1]")
    left_items = _validated_overlap_items(left, "left")
    right_items = left_items if right is None else _validated_overlap_items(right, "right")
    left_tokens = {
        item_id: normalize_overlap_text(item_text) for item_id, item_text in left_items
    }
    right_tokens = (
        left_tokens
        if right is None
        else {
            item_id: normalize_overlap_text(item_text)
            for item_id, item_text in right_items
        }
    )
    findings: list[NearDuplicateFinding] = []
    for left_index, (left_id, _left_text) in enumerate(left_items):
        start = left_index + 1 if right is None else 0
        for right_id, _right_text in right_items[start:]:
            similarity = _near_duplicate_similarity_tokens(
                left_tokens[left_id],
                right_tokens[right_id],
            )
            if similarity >= threshold:
                findings.append(NearDuplicateFinding(left_id, right_id, similarity))
    return tuple(sorted(findings, key=lambda item: (item.left_id, item.right_id)))


def create_document(
    *,
    text: str,
    source: str,
    document_id: str | None = None,
    fields: Mapping[str, str] | None = None,
    explicit_split: str | None = None,
) -> Document:
    canonical_text = normalize_document_text(text)
    if not canonical_text:
        raise DocumentValidationError("Document text must contain non-whitespace content.")
    logical_source = normalize_logical_name(source)
    normalized_fields: list[tuple[str, str]] = []
    for name, value in sorted((fields or {}).items()):
        if not isinstance(name, str) or not name.strip():
            raise DocumentValidationError("Document field names must be non-empty strings.")
        if not isinstance(value, str):
            raise DocumentValidationError(f"Document field {name!r} must be a string.")
        normalized_fields.append((name.strip(), value))
    if explicit_split not in {None, "train", "validation"}:
        raise DocumentValidationError(
            "Document explicit_split must be 'train', 'validation', or absent."
        )

    raw_payload = {"text": text, "fields": dict(normalized_fields)}
    canonical_payload = {
        "text": canonical_text,
        "fields": {
            name: normalize_document_text(value) for name, value in normalized_fields
        },
    }
    raw_fingerprint = hashlib.sha256(canonical_json_bytes(raw_payload)).hexdigest()
    canonical_fingerprint = canonical_sha256(canonical_payload)
    if document_id is None:
        document_id = "doc-" + canonical_sha256(
            {"source": logical_source, "content": canonical_payload}
        )[:24]
    if not isinstance(document_id, str) or not document_id.strip():
        raise DocumentValidationError("Document ID must be non-empty text.")
    clean_id = document_id.strip()
    if re.match(r"^[A-Za-z]:[\\/]", clean_id) or clean_id.startswith(("/", "\\")):
        raise DocumentValidationError("Document ID must not contain an absolute path.")
    return Document(
        document_id=clean_id,
        text=text,
        source=logical_source,
        fields=tuple(normalized_fields),
        explicit_split=explicit_split,
        raw_fingerprint=raw_fingerprint,
        canonical_fingerprint=canonical_fingerprint,
    )


def read_pretraining_documents(raw_data_dir: str | Path) -> list[Document]:
    """Read real Markdown source files and explicit top-level JSONL records."""

    raw_path = Path(raw_data_dir)
    legacy_combined = raw_path / "byteseed_personal_assistant_corpus.md"
    markdown_paths = [
        path
        for path in (
            *raw_path.glob("*.md"),
            *(raw_path / "personal_assistant").glob("*.md"),
            *(raw_path / "generated" / "markdown").glob("*.md"),
        )
        if path != legacy_combined
    ]
    paths = sorted(
        [*markdown_paths, *raw_path.glob("*.jsonl")],
        key=lambda path: path.relative_to(raw_path).as_posix(),
    )
    if not paths:
        raise FileNotFoundError(
            f"No supported Markdown sources or top-level JSONL documents found in {raw_path}."
        )
    documents: list[Document] = []
    for path in paths:
        logical_source = path.relative_to(raw_path).as_posix()
        if path.suffix.lower() == ".md":
            documents.append(
                create_document(
                    text=path.read_text(encoding="utf-8-sig"),
                    source=logical_source,
                )
            )
        else:
            documents.extend(read_document_jsonl(path, default_source=logical_source))
    return documents


def read_document_jsonl(
    path: str | Path, *, default_source: str | None = None
) -> list[Document]:
    input_path = Path(path)
    documents: list[Document] = []
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DocumentValidationError(
                    f"Invalid JSON document record in {input_path.name}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise DocumentValidationError(
                    f"Document record {input_path.name}:{line_number} must be an object."
                )
            if "text" in row:
                text = row["text"]
                fields: dict[str, str] = {}
                if "user" in row or "assistant" in row or "system" in row:
                    _, fields = _conversation_record_text(
                        row, input_path.name, line_number
                    )
            else:
                text, fields = _conversation_record_text(row, input_path.name, line_number)
            if not isinstance(text, str):
                raise DocumentValidationError(
                    f"Document record {input_path.name}:{line_number} text must be a string."
                )
            source = row.get("source")
            if source is None:
                source = default_source or input_path.name
            if not isinstance(source, str):
                raise DocumentValidationError(
                    f"Document record {input_path.name}:{line_number} source must be text."
                )
            document_id = row.get("id")
            explicit_split = row.get("split")
            documents.append(
                create_document(
                    text=text,
                    source=source,
                    document_id=document_id,
                    fields=fields,
                    explicit_split=explicit_split,
                )
            )
    if not documents:
        raise DocumentValidationError(f"No document records found in {input_path.name}.")
    return documents


def read_sft_documents(path: str | Path) -> list[Document]:
    """Read SFT examples as bounded conversation documents without rewriting them."""

    input_path = Path(path)
    documents: list[Document] = []
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DocumentValidationError(
                    f"Invalid SFT JSON in {input_path.name}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise DocumentValidationError(
                    f"SFT record {input_path.name}:{line_number} must be an object."
                )
            text, fields = _conversation_record_text(row, input_path.name, line_number)
            source = row.get("source", input_path.name)
            if not isinstance(source, str):
                raise DocumentValidationError(
                    f"SFT record {input_path.name}:{line_number} source must be text."
                )
            document_id = row.get("id")
            documents.append(
                create_document(
                    text=text,
                    source=source,
                    document_id=document_id,
                    fields=fields,
                )
            )
    if not documents:
        raise DocumentValidationError(f"No SFT records found in {input_path.name}.")
    return documents


def deduplicate_documents(documents: Iterable[Document]) -> DuplicateAnalysis:
    ordered = tuple(sorted(documents, key=_document_sort_key))
    if not ordered:
        raise DocumentValidationError("At least one document is required.")
    identities: dict[str, set[str]] = defaultdict(set)
    for document in ordered:
        identities[document.document_id].add(document.canonical_fingerprint)
    ambiguous_ids = sorted(key for key, values in identities.items() if len(values) > 1)
    if ambiguous_ids:
        raise DocumentValidationError(
            "Document ID identifies different canonical content: " + ", ".join(ambiguous_ids)
        )

    canonical_groups: dict[str, list[Document]] = defaultdict(list)
    raw_groups: dict[str, list[Document]] = defaultdict(list)
    for document in ordered:
        canonical_groups[document.canonical_fingerprint].append(document)
        raw_groups[document.raw_fingerprint].append(document)

    groups: list[DuplicateGroup] = []
    representatives: list[Document] = []
    for fingerprint in sorted(canonical_groups):
        members = tuple(sorted(canonical_groups[fingerprint], key=_document_sort_key))
        explicit = {member.explicit_split for member in members if member.explicit_split}
        if len(explicit) > 1:
            raise DocumentValidationError(
                "Canonical duplicate group has conflicting explicit splits for document IDs: "
                + ", ".join(member.document_id for member in members)
            )
        representative = members[0]
        groups.append(
            DuplicateGroup(
                canonical_fingerprint=fingerprint,
                representative=representative,
                members=members,
                removed=members[1:],
            )
        )
        representatives.append(representative)

    raw_duplicate_count = sum(len(group) - 1 for group in raw_groups.values())
    removed_count = len(ordered) - len(groups)
    return DuplicateAnalysis(
        groups=tuple(groups),
        representatives=tuple(sorted(representatives, key=_document_sort_key)),
        raw_duplicate_count=raw_duplicate_count,
        canonical_duplicate_count=removed_count - raw_duplicate_count,
    )


def split_duplicate_groups(
    analysis: DuplicateAnalysis,
    *,
    seed: int,
    validation_ratio: float,
) -> tuple[tuple[Document, ...], tuple[Document, ...], dict[str, str]]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Split seed must be an integer.")
    if (
        isinstance(validation_ratio, bool)
        or not isinstance(validation_ratio, (int, float))
        or not 0 < float(validation_ratio) < 1
    ):
        raise ValueError("validation_ratio must be numeric and between 0 and 1.")
    if len(analysis.groups) < 2:
        raise DataQualityError(
            "Document-aware splitting requires at least two unique duplicate groups."
        )

    scores: dict[str, float] = {}
    assignments: dict[str, str] = {}
    for group in analysis.groups:
        explicit = {
            member.explicit_split for member in group.members if member.explicit_split
        }
        if explicit:
            assignments[group.canonical_fingerprint] = next(iter(explicit))
            continue
        score = _split_score(group.canonical_fingerprint, seed)
        scores[group.canonical_fingerprint] = score
        assignments[group.canonical_fingerprint] = (
            "validation" if score < float(validation_ratio) else "train"
        )

    if "validation" not in assignments.values():
        candidates = [
            group for group in analysis.groups if not any(member.explicit_split for member in group.members)
        ]
        if not candidates:
            raise DataQualityError("Explicit split assignments would produce an empty validation split.")
        selected = min(candidates, key=lambda group: (scores[group.canonical_fingerprint], group.canonical_fingerprint))
        assignments[selected.canonical_fingerprint] = "validation"
    if "train" not in assignments.values():
        candidates = [
            group for group in analysis.groups if not any(member.explicit_split for member in group.members)
        ]
        if not candidates:
            raise DataQualityError("Explicit split assignments would produce an empty training split.")
        selected = max(candidates, key=lambda group: (scores[group.canonical_fingerprint], group.canonical_fingerprint))
        assignments[selected.canonical_fingerprint] = "train"

    train = tuple(
        sorted(
            (
                group.representative
                for group in analysis.groups
                if assignments[group.canonical_fingerprint] == "train"
            ),
            key=_document_sort_key,
        )
    )
    validation = tuple(
        sorted(
            (
                group.representative
                for group in analysis.groups
                if assignments[group.canonical_fingerprint] == "validation"
            ),
            key=_document_sort_key,
        )
    )
    validate_no_leakage(train, validation)
    return train, validation, assignments


def validate_no_leakage(
    train_documents: Iterable[Document], validation_documents: Iterable[Document]
) -> None:
    train = tuple(train_documents)
    validation = tuple(validation_documents)
    checks = (
        ("document ID", {item.document_id for item in train}, {item.document_id for item in validation}),
        ("raw fingerprint", {item.raw_fingerprint for item in train}, {item.raw_fingerprint for item in validation}),
        (
            "canonical fingerprint",
            {item.canonical_fingerprint for item in train},
            {item.canonical_fingerprint for item in validation},
        ),
    )
    for label, train_values, validation_values in checks:
        overlap = sorted(train_values & validation_values)
        if overlap:
            values = ", ".join(value[:12] if len(value) == 64 else value for value in overlap)
            raise DataLeakageError(
                f"Cross-split {label} leakage between train and validation: {values}."
            )


def detect_evaluation_contamination(
    documents: Iterable[Document],
    *,
    prompts: Iterable[EvaluationPrompt] | None = None,
    dataset_split: str = "pre-split",
) -> tuple[ContaminationFinding, ...]:
    prompt_records = tuple(
        registered_evaluation_prompts() if prompts is None else prompts
    )
    prompt_ids = [prompt.prompt_id for prompt in prompt_records]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise DocumentValidationError("Registered evaluation prompt IDs must be unique.")
    findings: dict[tuple[str, str, str], ContaminationFinding] = {}
    for document in sorted(documents, key=_document_sort_key):
        document_text = normalize_document_text(document.text)
        field_texts = {
            name: normalize_document_text(value) for name, value in document.fields
        }
        for prompt in prompt_records:
            normalized_prompt = normalize_document_text(prompt.text)
            match_type: str | None = None
            if document_text == normalized_prompt:
                match_type = "full-document"
            elif any(value == normalized_prompt for value in field_texts.values()):
                match_type = "field-exact"
            elif normalized_prompt in document_text or any(
                normalized_prompt in value for value in field_texts.values()
            ):
                match_type = "normalized-substring"
            if match_type is not None:
                key = (prompt.prompt_id, document.document_id, document.source)
                findings[key] = ContaminationFinding(
                    prompt_id=prompt.prompt_id,
                    suite=prompt.suite,
                    dataset_split=dataset_split,
                    document_id=document.document_id,
                    source=document.source,
                    match_type=match_type,
                    document_fingerprint=document.canonical_fingerprint[:12],
                )
    return tuple(
        sorted(
            findings.values(),
            key=lambda item: (item.suite, item.prompt_id, item.document_id, item.source),
        )
    )


def plan_document_dataset(
    documents: Iterable[Document],
    *,
    seed: int,
    validation_ratio: float,
    allow_eval_contamination: bool = False,
    prompts: Iterable[EvaluationPrompt] | None = None,
) -> DocumentPlan:
    source_documents = tuple(sorted(documents, key=_document_sort_key))
    prompt_records = tuple(
        registered_evaluation_prompts() if prompts is None else prompts
    )
    duplicates = deduplicate_documents(source_documents)
    pre_split_findings = detect_evaluation_contamination(
        source_documents, prompts=prompt_records, dataset_split="pre-split"
    )
    if pre_split_findings and not allow_eval_contamination:
        raise EvaluationContaminationError(pre_split_findings)
    if pre_split_findings:
        warnings.warn(
            "Evaluation contamination explicitly accepted for historical reproduction; "
            "affected results are not held-out generalization.",
            UserWarning,
            stacklevel=2,
        )
    train, validation, assignments = split_duplicate_groups(
        duplicates, seed=seed, validation_ratio=validation_ratio
    )
    source_by_key = {
        (document.document_id, document.source): document for document in source_documents
    }
    findings = tuple(
        replace(
            finding,
            dataset_split=assignments[
                source_by_key[(finding.document_id, finding.source)].canonical_fingerprint
            ],
        )
        for finding in pre_split_findings
    )
    return DocumentPlan(
        documents=source_documents,
        duplicates=duplicates,
        train_documents=train,
        validation_documents=validation,
        findings=findings,
        audited_prompts=prompt_records,
        assignments=dict(sorted(assignments.items())),
        split_seed=seed,
        validation_ratio=float(validation_ratio),
        allow_eval_contamination=bool(allow_eval_contamination),
    )


def build_data_quality_report(
    plan: DocumentPlan,
    *,
    train_token_count: int | None = None,
    validation_token_count: int | None = None,
) -> dict[str, Any]:
    if (train_token_count is None) != (validation_token_count is None):
        raise ValueError("Train and validation token counts must be provided together.")
    removed_groups = []
    for group in plan.duplicates.groups:
        if not group.removed:
            continue
        removed_groups.append(
            {
                "canonical_fingerprint": group.canonical_fingerprint[:12],
                "representative": {
                    "document_id": group.representative.document_id,
                    "source": group.representative.source,
                },
                "removed": [
                    {"document_id": item.document_id, "source": item.source}
                    for item in group.removed
                ],
            }
        )
    suite_versions = {
        suite.suite_id: suite.version for suite in registered_evaluation_suites()
    }
    suite_order: list[str] = []
    prompt_ids_by_suite: dict[str, list[str]] = {}
    for prompt in plan.audited_prompts:
        if prompt.suite not in prompt_ids_by_suite:
            suite_order.append(prompt.suite)
            prompt_ids_by_suite[prompt.suite] = []
        prompt_ids_by_suite[prompt.suite].append(prompt.prompt_id)
    audit_suites = [
        {
            "suite_id": suite_id,
            "suite_version": suite_versions.get(suite_id, 1),
            "prompt_ids": prompt_ids_by_suite[suite_id],
        }
        for suite_id in suite_order
    ]
    report: dict[str, Any] = {
        "version": DATA_QUALITY_REPORT_VERSION,
        "algorithm": HASH_ALGORITHM,
        "normalization_version": NORMALIZATION_VERSION,
        "document_format_version": DOCUMENT_FORMAT_VERSION,
        "deduplication": {
            "version": DEDUPLICATION_VERSION,
            "strategy": DEDUPLICATION_STRATEGY,
            "representative_order": "document-id,source,raw-fingerprint",
        },
        "split": {
            "version": SPLIT_STRATEGY_VERSION,
            "strategy": SPLIT_STRATEGY,
            "seed": plan.split_seed,
            "validation_ratio": plan.validation_ratio,
        },
        "counts": {
            "input_documents": len(plan.documents),
            "accepted_unique_documents": len(plan.duplicates.representatives),
            "raw_duplicates": plan.duplicates.raw_duplicate_count,
            "canonical_duplicates": plan.duplicates.canonical_duplicate_count,
            "removed_duplicates": plan.duplicates.removed_count,
            "train_documents": len(plan.train_documents),
            "validation_documents": len(plan.validation_documents),
            "contamination_matches": len(plan.findings),
        },
        "removed_duplicate_groups": removed_groups,
        "contamination_findings": [finding.as_dict() for finding in plan.findings],
        "evaluation_audit": {
            "registry_version": EVALUATION_PROMPT_REGISTRY_VERSION,
            "suites": audit_suites,
        },
        "leakage_validation": "passed",
        "policy": {
            "allow_eval_contamination": plan.allow_eval_contamination,
            "held_out_generalization_measured": False,
        },
    }
    if train_token_count is not None and validation_token_count is not None:
        if train_token_count < 0 or validation_token_count < 0:
            raise ValueError("Token counts must be non-negative.")
        report["counts"]["train_tokens"] = int(train_token_count)
        report["counts"]["validation_tokens"] = int(validation_token_count)
    report["digest"] = canonical_sha256(_quality_report_digest_payload(report))
    validate_data_quality_report(report)
    return report


def validate_data_quality_report(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise ProvenanceValidationError("Data-quality report must be a mapping.")
    if report.get("version") != DATA_QUALITY_REPORT_VERSION:
        raise ProvenanceValidationError(
            f"Unsupported data-quality report version {report.get('version')!r}; "
            f"this ByteSeed build supports version {DATA_QUALITY_REPORT_VERSION}."
        )
    if report.get("algorithm") != HASH_ALGORITHM:
        raise ProvenanceValidationError("Data-quality report hash algorithm must be sha256.")
    if report.get("normalization_version") != NORMALIZATION_VERSION:
        raise ProvenanceValidationError("Unsupported data-quality normalization version.")
    if report.get("document_format_version") != DOCUMENT_FORMAT_VERSION:
        raise ProvenanceValidationError("Unsupported document format version.")
    deduplication = report.get("deduplication")
    split = report.get("split")
    counts = report.get("counts")
    policy = report.get("policy")
    if not isinstance(deduplication, Mapping) or deduplication.get("version") != DEDUPLICATION_VERSION:
        raise ProvenanceValidationError("Data-quality deduplication policy is malformed.")
    if deduplication.get("strategy") != DEDUPLICATION_STRATEGY:
        raise ProvenanceValidationError("Unsupported data-quality deduplication strategy.")
    if (
        not isinstance(deduplication.get("representative_order"), str)
        or not deduplication["representative_order"].strip()
    ):
        raise ProvenanceValidationError(
            "Data-quality representative ordering must be non-empty text."
        )
    if not isinstance(split, Mapping) or split.get("version") != SPLIT_STRATEGY_VERSION:
        raise ProvenanceValidationError("Data-quality split policy is malformed.")
    if split.get("strategy") != SPLIT_STRATEGY:
        raise ProvenanceValidationError("Unsupported data-quality split strategy.")
    seed = split.get("seed")
    ratio = split.get("validation_ratio")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProvenanceValidationError("Data-quality split seed must be an integer.")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not 0 < float(ratio) < 1
    ):
        raise ProvenanceValidationError(
            "Data-quality validation_ratio must be numeric and between 0 and 1."
        )
    if not isinstance(counts, Mapping) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise ProvenanceValidationError("Data-quality report counts must be non-negative integers.")
    if not isinstance(policy, Mapping) or not isinstance(
        policy.get("allow_eval_contamination"), bool
    ):
        raise ProvenanceValidationError("Data-quality contamination policy is malformed.")
    if policy.get("held_out_generalization_measured") is not False:
        raise ProvenanceValidationError(
            "PR 6 data-quality reports must not claim held-out generalization."
        )
    if report.get("leakage_validation") != "passed":
        raise ProvenanceValidationError(
            "Data-quality report must record passed leakage validation."
        )
    audit = report.get("evaluation_audit")
    audited_pairs: set[tuple[str, str]] | None = None
    if audit is not None:
        if (
            not isinstance(audit, Mapping)
            or audit.get("registry_version") != EVALUATION_PROMPT_REGISTRY_VERSION
            or not isinstance(audit.get("suites"), list)
        ):
            raise ProvenanceValidationError(
                "Data-quality evaluation audit metadata is malformed."
            )
        audited_pairs = set()
        suite_ids: set[str] = set()
        for suite in audit["suites"]:
            if not isinstance(suite, Mapping) or set(suite) != {
                "suite_id",
                "suite_version",
                "prompt_ids",
            }:
                raise ProvenanceValidationError(
                    "Data-quality evaluation audit suite metadata is malformed."
                )
            suite_id = suite["suite_id"]
            suite_version = suite["suite_version"]
            prompt_ids = suite["prompt_ids"]
            if (
                not isinstance(suite_id, str)
                or not suite_id.strip()
                or suite_id in suite_ids
                or suite_version != 1
                or not isinstance(prompt_ids, list)
                or not all(
                    isinstance(prompt_id, str) and prompt_id.strip()
                    for prompt_id in prompt_ids
                )
                or len(prompt_ids) != len(set(prompt_ids))
            ):
                raise ProvenanceValidationError(
                    "Data-quality evaluation audit suite identity is invalid."
                )
            suite_ids.add(suite_id)
            for prompt_id in prompt_ids:
                pair = (suite_id, prompt_id)
                if pair in audited_pairs:
                    raise ProvenanceValidationError(
                        "Data-quality evaluation audit contains duplicate prompt identity."
                    )
                audited_pairs.add(pair)
    findings = report.get("contamination_findings")
    removed = report.get("removed_duplicate_groups")
    if not isinstance(findings, list) or not isinstance(removed, list):
        raise ProvenanceValidationError("Data-quality report findings must be lists.")
    if counts.get("contamination_matches") != len(findings):
        raise ProvenanceValidationError(
            "Data-quality contamination count does not match its findings."
        )
    if counts.get("accepted_unique_documents", -1) + counts.get(
        "removed_duplicates", -1
    ) != counts.get("input_documents"):
        raise ProvenanceValidationError(
            "Data-quality input, unique, and removed counts are inconsistent."
        )
    if counts.get("train_documents", -1) + counts.get(
        "validation_documents", -1
    ) != counts.get("accepted_unique_documents"):
        raise ProvenanceValidationError(
            "Data-quality train/validation counts do not cover unique documents."
        )
    if counts.get("raw_duplicates", -1) + counts.get(
        "canonical_duplicates", -1
    ) != counts.get("removed_duplicates"):
        raise ProvenanceValidationError(
            "Data-quality duplicate counts are inconsistent."
        )
    for finding in findings:
        if not isinstance(finding, Mapping):
            raise ProvenanceValidationError(
                "Data-quality contamination findings must be mappings."
            )
        required = {
            "prompt_id",
            "suite",
            "dataset_split",
            "document_id",
            "source",
            "match_type",
            "document_fingerprint",
        }
        if not required.issubset(finding):
            raise ProvenanceValidationError(
                "Data-quality contamination finding is missing required identity fields."
            )
        if not all(
            isinstance(finding[field], str) and finding[field]
            for field in required
        ):
            raise ProvenanceValidationError(
                "Data-quality contamination identity fields must be non-empty text."
            )
        if audited_pairs is not None and (
            finding["suite"], finding["prompt_id"]
        ) not in audited_pairs:
            raise ProvenanceValidationError(
                "Data-quality contamination finding is outside recorded audit coverage."
            )
        if finding["dataset_split"] not in {"train", "validation"}:
            raise ProvenanceValidationError(
                "Data-quality contamination dataset_split must be train or validation."
            )
        normalize_logical_name(finding["source"])
        if finding["match_type"] not in {
            "full-document",
            "field-exact",
            "normalized-substring",
        }:
            raise ProvenanceValidationError(
                "Data-quality contamination match_type is unsupported."
            )
        if re.fullmatch(r"[0-9a-f]{12}", finding["document_fingerprint"]) is None:
            raise ProvenanceValidationError(
                "Data-quality finding fingerprint must be 12 lowercase hex characters."
            )
    digest = report.get("digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ProvenanceValidationError("Data-quality report digest must be lowercase SHA-256.")
    if digest != canonical_sha256(_quality_report_digest_payload(report)):
        raise ProvenanceValidationError("Data-quality report digest does not match its identity fields.")


def data_quality_preprocessing_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return manifest-v2 preprocessing identity from a validated report."""

    validate_data_quality_report(report)
    split = report["split"]
    deduplication = report["deduplication"]
    policy = report["policy"]
    return {
        "version": 2,
        "builder": "byteseed.prepare_data",
        "document_format": {
            "version": report["document_format_version"],
            "name": "markdown-source-or-jsonl-record",
        },
        "tokenization": {
            "add_bos": True,
            "add_eos": True,
            "per_document": True,
        },
        "normalization": {"version": report["normalization_version"]},
        "deduplication": {
            "version": deduplication["version"],
            "strategy": deduplication["strategy"],
            "representative_order": deduplication["representative_order"],
        },
        "split": {
            "version": split["version"],
            "strategy": split["strategy"],
            "seed": split["seed"],
            "validation_ratio": split["validation_ratio"],
        },
        "data_quality": {
            "report_version": report["version"],
            "report_digest": report["digest"],
            "allow_eval_contamination": policy["allow_eval_contamination"],
        },
    }


def write_data_quality_report(path: str | Path, report: Mapping[str, Any]) -> None:
    validate_data_quality_report(report)
    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def audit_sft_file(
    path: str | Path,
    *,
    prompts: Iterable[EvaluationPrompt] | None = None,
) -> dict[str, Any]:
    """Return a deterministic, non-writing audit of one SFT JSONL file."""

    documents = read_sft_documents(path)
    duplicates = deduplicate_documents(documents)
    findings = detect_evaluation_contamination(documents, prompts=prompts)
    report: dict[str, Any] = {
        "version": DATA_QUALITY_REPORT_VERSION,
        "kind": "sft-audit",
        "algorithm": HASH_ALGORITHM,
        "normalization_version": NORMALIZATION_VERSION,
        "source": Path(path).name,
        "counts": {
            "input_documents": len(documents),
            "accepted_unique_documents": len(duplicates.representatives),
            "raw_duplicates": duplicates.raw_duplicate_count,
            "canonical_duplicates": duplicates.canonical_duplicate_count,
            "removed_duplicates": duplicates.removed_count,
            "contamination_matches": len(findings),
        },
        "contamination_findings": [finding.as_dict() for finding in findings],
        "held_out_generalization_measured": False,
    }
    report["digest"] = canonical_sha256({key: value for key, value in report.items() if key != "digest"})
    return report


def _conversation_record_text(
    row: Mapping[str, Any], source_name: str, line_number: int
) -> tuple[str, dict[str, str]]:
    fields: dict[str, str] = {}
    system = row.get("system")
    if system is not None:
        if not isinstance(system, str) or not system.strip():
            raise DocumentValidationError(
                f"SFT record {source_name}:{line_number} system must be non-empty text when present."
            )
        fields["system"] = system
    for required in ("user", "assistant"):
        value = row.get(required)
        if not isinstance(value, str) or not value.strip():
            raise DocumentValidationError(
                f"SFT record {source_name}:{line_number} requires non-empty {required} text."
            )
        fields[required] = value
    parts: list[str] = []
    if "system" in fields:
        parts.extend(("<|system|>", fields["system"]))
    parts.extend(
        (
            "<|user|>",
            fields["user"],
            "<|assistant|>",
            fields["assistant"],
            "<|end|>",
        )
    )
    return "\n".join(parts), fields


def _singular_overlap_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def _validated_overlap_items(
    items: Iterable[tuple[str, str]],
    label: str,
) -> tuple[tuple[str, str], ...]:
    validated: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{label} near-duplicate items must be (ID, text) tuples")
        item_id, text = item
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"{label} near-duplicate item IDs must be non-empty text")
        if item_id in seen:
            raise ValueError(f"duplicate {label} near-duplicate item ID: {item_id!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label} near-duplicate text must be non-empty")
        seen.add(item_id)
        validated.append((item_id, text))
    return tuple(sorted(validated))


def _document_sort_key(document: Document) -> tuple[str, str, str]:
    return document.document_id, document.source, document.raw_fingerprint


def _split_score(canonical_fingerprint: str, seed: int) -> float:
    payload = f"{SPLIT_STRATEGY_VERSION}\0{seed}\0{canonical_fingerprint}".encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return value / float(1 << 256)


def _quality_report_digest_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "digest"}
