from __future__ import annotations

import copy
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from byteseed.data_quality import (
    DATA_QUALITY_REPORT_VERSION,
    DataLeakageError,
    DataQualityError,
    build_data_quality_report,
    create_document,
    data_quality_preprocessing_identity,
    deduplicate_documents,
    plan_document_dataset,
    read_pretraining_documents,
    split_duplicate_groups,
    validate_data_quality_report,
    validate_no_leakage,
)
from byteseed.prepare_data import prepare_document_arrays
from byteseed.provenance import ProvenanceValidationError, canonical_sha256


def _documents(count: int = 12):
    return [
        create_document(
            text=f"Document content {index}",
            source=f"logical/{index}.md",
            document_id=f"doc-{index:02d}",
        )
        for index in range(count)
    ]


def _plan(documents, *, seed=17, validation_ratio=0.25):
    return plan_document_dataset(
        documents,
        seed=seed,
        validation_ratio=validation_ratio,
        prompts=(),
    )


def test_document_split_is_complete_disjoint_and_order_independent():
    documents = _documents()

    forward = _plan(documents)
    reverse = _plan(reversed(documents))

    assert forward.assignments == reverse.assignments
    assert {item.document_id for item in forward.train_documents}.isdisjoint(
        item.document_id for item in forward.validation_documents
    )
    assert len(forward.train_documents) + len(forward.validation_documents) == len(
        documents
    )


def test_split_seed_changes_assignment_on_suitable_fixture():
    documents = _documents(40)

    first = _plan(documents, seed=1)
    second = _plan(documents, seed=2)

    assert first.assignments != second.assignments


def test_validation_ratio_and_seed_participate_in_report_identity():
    documents = _documents(40)
    first = build_data_quality_report(_plan(documents, seed=3, validation_ratio=0.2))
    changed_seed = build_data_quality_report(
        _plan(documents, seed=4, validation_ratio=0.2)
    )
    changed_ratio = build_data_quality_report(
        _plan(documents, seed=3, validation_ratio=0.4)
    )

    assert len({first["digest"], changed_seed["digest"], changed_ratio["digest"]}) == 3


def test_duplicate_group_members_share_one_assignment():
    duplicate_a = create_document(
        text="same content", source="a.md", document_id="a"
    )
    duplicate_b = create_document(
        text="same  content", source="b.md", document_id="b"
    )
    analysis = deduplicate_documents([duplicate_a, duplicate_b, *_documents(4)])

    train, validation, assignments = split_duplicate_groups(
        analysis, seed=7, validation_ratio=0.5
    )

    assert duplicate_a.canonical_fingerprint == duplicate_b.canonical_fingerprint
    assert assignments[duplicate_a.canonical_fingerprint] in {"train", "validation"}
    assert sum(item.canonical_fingerprint == duplicate_a.canonical_fingerprint for item in (*train, *validation)) == 1


def test_tiny_impossible_unique_group_fails_clearly():
    analysis = deduplicate_documents(
        [
            create_document(text="same", source="a.md", document_id="a"),
            create_document(text="same", source="b.md", document_id="b"),
        ]
    )

    with pytest.raises(DataQualityError, match="at least two unique"):
        split_duplicate_groups(analysis, seed=1, validation_ratio=0.2)


def test_explicit_assignments_cannot_silently_empty_training_split():
    analysis = deduplicate_documents(
        [
            create_document(
                text=f"document {index}",
                source=f"{index}.md",
                document_id=str(index),
                explicit_split="validation",
            )
            for index in range(3)
        ]
    )

    with pytest.raises(DataQualityError, match="empty training split"):
        split_duplicate_groups(analysis, seed=1, validation_ratio=0.2)


def test_leakage_validator_rejects_id_raw_and_canonical_overlap():
    train = create_document(text="alpha", source="train.md", document_id="same-id")
    same_id = create_document(text="beta", source="val.md", document_id="same-id")
    with pytest.raises(DataLeakageError, match="document ID"):
        validate_no_leakage([train], [same_id])

    same_raw = create_document(text="alpha", source="val.md", document_id="other")
    with pytest.raises(DataLeakageError, match="raw fingerprint"):
        validate_no_leakage([train], [same_raw])

    canonical = create_document(text="alpha  ", source="val.md", document_id="third")
    with pytest.raises(DataLeakageError, match="canonical fingerprint"):
        validate_no_leakage([train], [canonical])


def test_clean_split_passes_leakage_validation():
    train = _documents(2)
    validation = [
        create_document(
            text=f"validation content {index}",
            source=f"validation/{index}.md",
            document_id=f"validation-{index}",
        )
        for index in range(2)
    ]

    validate_no_leakage(train, validation)


def test_top_level_markdown_and_jsonl_records_are_real_boundaries(tmp_path):
    (tmp_path / "b.md").write_text("second", encoding="utf-8")
    (tmp_path / "a.md").write_text("first", encoding="utf-8")
    (tmp_path / "records.jsonl").write_text(
        json.dumps({"id": "json-doc", "text": "third"}) + "\n",
        encoding="utf-8",
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.md").write_text("not an implicit boundary", encoding="utf-8")
    personal = tmp_path / "personal_assistant"
    personal.mkdir()
    (personal / "note.md").write_text("personal source", encoding="utf-8")
    (tmp_path / "byteseed_personal_assistant_corpus.md").write_text(
        "legacy combined output", encoding="utf-8"
    )

    documents = read_pretraining_documents(tmp_path)

    assert len(documents) == 4
    assert {item.text for item in documents} == {
        "first",
        "second",
        "third",
        "personal source",
    }
    assert "personal_assistant/note.md" in {item.source for item in documents}


class _FakeTokenizer:
    def __init__(self):
        self.calls: list[tuple[str, bool, bool]] = []

    def encode(self, text: str, *, add_bos: bool, add_eos: bool):
        self.calls.append((text, add_bos, add_eos))
        value = int(text.rsplit(" ", 1)[-1]) + 10
        return [1, value, 2]


def test_preparation_splits_documents_before_tokenization_and_preserves_boundaries():
    documents = _documents(8)
    tokenizer = _FakeTokenizer()

    train, validation, report = prepare_document_arrays(
        documents,
        tokenizer,
        train_split=0.75,
        split_seed=9,
        vocab_size=100,
    )

    assert len(tokenizer.calls) == len(documents)
    assert all(add_bos and add_eos for _, add_bos, add_eos in tokenizer.calls)
    assert train.dtype == np.uint16
    assert validation.dtype == np.uint16
    assert train.tolist().count(1) == report["counts"]["train_documents"]
    assert validation.tolist().count(1) == report["counts"]["validation_documents"]
    assert train.tolist().count(2) == report["counts"]["train_documents"]
    assert validation.tolist().count(2) == report["counts"]["validation_documents"]
    assert set(train[1::3]).isdisjoint(set(validation[1::3]))


def test_prepare_data_writes_v2_manifest_and_quality_report_under_tmp_path(
    tmp_path, monkeypatch, tokenizer_identity
):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    for index in range(6):
        (raw / f"{index}.md").write_text(
            f"Document content {index}", encoding="utf-8"
        )
    tokenizer = _FakeTokenizer()
    tokenizer.identity = tokenizer_identity
    module = importlib.import_module("byteseed.prepare_data")
    config = SimpleNamespace(
        raw_data_dir=raw,
        processed_data_dir=processed,
        tokenizer_dir=tmp_path / "tokenizer",
        train_split=0.75,
        seed=9,
        vocab_size=100,
        block_size=4,
    )
    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(module, "ByteSeedTokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(module, "warn_if_tiny_dataset", lambda *_args: None)

    module.prepare_data("synthetic.yaml")

    manifest = json.loads(
        (processed / "data_manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (processed / "data_quality_report.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == 2
    assert manifest["preprocessing"]["data_quality"]["report_digest"] == report["digest"]
    assert np.load(processed / "train.npy", allow_pickle=False).ndim == 1
    assert np.load(processed / "val.npy", allow_pickle=False).ndim == 1


def test_quality_report_is_order_independent_and_omits_machine_paths(tmp_path):
    documents = _documents()
    first = build_data_quality_report(_plan(documents))
    second = build_data_quality_report(_plan(reversed(documents)))

    assert first["digest"] == second["digest"]
    serialized = json.dumps(first, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "timestamp" not in serialized.lower()
    assert first["policy"]["held_out_generalization_measured"] is False


def test_quality_report_future_version_and_malformed_counts_fail():
    report = build_data_quality_report(_plan(_documents()))
    future = dict(report, version=DATA_QUALITY_REPORT_VERSION + 1)
    with pytest.raises(ProvenanceValidationError, match="Unsupported data-quality report"):
        validate_data_quality_report(future)

    malformed = copy.deepcopy(report)
    malformed["counts"]["train_documents"] = -1
    malformed["digest"] = canonical_sha256(
        {key: value for key, value in malformed.items() if key != "digest"}
    )
    with pytest.raises(ProvenanceValidationError, match="non-negative"):
        validate_data_quality_report(malformed)


def test_report_identity_exports_document_aware_manifest_policy():
    report = build_data_quality_report(_plan(_documents()))

    identity = data_quality_preprocessing_identity(report)

    assert identity["version"] == 2
    assert identity["split"]["strategy"] == "canonical-group-sha256"
    assert identity["data_quality"]["report_digest"] == report["digest"]
    assert identity["tokenization"]["per_document"] is True
