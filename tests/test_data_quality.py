from __future__ import annotations

import json

import pytest

from byteseed.data_quality import (
    NORMALIZATION_VERSION,
    DocumentValidationError,
    audit_sft_file,
    create_document,
    deduplicate_documents,
    normalize_document_text,
    read_document_jsonl,
    read_sft_documents,
)


def _document(
    text: str,
    document_id: str,
    *,
    source: str = "fixture.md",
    split: str | None = None,
):
    return create_document(
        text=text,
        source=source,
        document_id=document_id,
        explicit_split=split,
    )


def test_explicit_document_id_is_preserved():
    document = _document("alpha", "source-stable-id")

    assert document.document_id == "source-stable-id"


def test_derived_identity_ignores_machine_directory_and_input_order(tmp_path):
    first = tmp_path / "one" / "records.jsonl"
    second = tmp_path / "two" / "records.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    rows = [
        {"source": "logical/alpha", "text": "Alpha"},
        {"source": "logical/beta", "text": "Beta"},
    ]
    first.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    second.write_text(
        "\n".join(json.dumps(row) for row in reversed(rows)), encoding="utf-8"
    )

    first_documents = read_document_jsonl(first)
    second_documents = read_document_jsonl(second)

    assert {item.document_id for item in first_documents} == {
        item.document_id for item in second_documents
    }
    assert all(str(tmp_path) not in item.document_id for item in first_documents)


def test_meaningful_content_change_changes_canonical_identity():
    first = _document("one value", "first")
    second = _document("two values", "second")

    assert first.canonical_fingerprint != second.canonical_fingerprint


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Alpha\r\nBeta", "Alpha\nBeta"),
        ("Cafe\u0301", "Caf\u00e9"),
        ("\n\n Alpha \n", "Alpha"),
        ("Alpha   beta", "Alpha beta"),
    ],
)
def test_conservative_normalization_equivalences(first, second):
    assert normalize_document_text(first) == normalize_document_text(second)


def test_normalization_preserves_fenced_code_spacing():
    text = "Before\n```python\nvalue  =  1\n```\nAfter"

    normalized = normalize_document_text(text)

    assert "value  =  1" in normalized
    assert NORMALIZATION_VERSION == 1


def test_exact_and_canonical_duplicates_are_counted_separately():
    analysis = deduplicate_documents(
        [
            _document("Alpha  beta\r\n", "b"),
            _document("Alpha  beta\r\n", "a"),
            _document("Alpha beta\n", "c"),
            _document("Distinct", "d"),
        ]
    )

    assert analysis.raw_duplicate_count == 1
    assert analysis.canonical_duplicate_count == 1
    assert analysis.removed_count == 2
    assert [item.document_id for item in analysis.representatives] == ["a", "d"]


def test_duplicate_representative_is_independent_of_input_order():
    documents = [_document("same", "z"), _document("same", "a")]

    forward = deduplicate_documents(documents)
    reverse = deduplicate_documents(reversed(documents))

    assert forward.representatives == reverse.representatives
    assert forward.representatives[0].document_id == "a"
    assert forward.groups[0].removed[0].document_id == "z"


def test_conflicting_duplicate_explicit_splits_fail_clearly():
    with pytest.raises(DocumentValidationError, match="conflicting explicit splits"):
        deduplicate_documents(
            [
                _document("same", "a", split="train"),
                _document("same", "b", split="validation"),
            ]
        )


def test_one_document_id_cannot_name_different_content():
    with pytest.raises(DocumentValidationError, match="different canonical content"):
        deduplicate_documents(
            [_document("alpha", "same-id"), _document("beta", "same-id")]
        )


def test_sft_reader_preserves_each_conversation_as_a_document(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps({"id": "chat-1", "user": "Question", "assistant": "Answer"})
        + "\n",
        encoding="utf-8",
    )

    documents = read_sft_documents(path)

    assert len(documents) == 1
    assert documents[0].document_id == "chat-1"
    assert documents[0].field_map() == {"assistant": "Answer", "user": "Question"}
    assert "<|user|>\nQuestion\n<|assistant|>\nAnswer\n<|end|>" == documents[0].text


@pytest.mark.parametrize(
    "row",
    [
        {"user": "", "assistant": "answer"},
        {"user": "question", "assistant": ""},
        {"user": "question"},
        ["not", "an", "object"],
    ],
)
def test_malformed_or_empty_sft_records_fail(tmp_path, row):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(DocumentValidationError):
        read_sft_documents(path)


def test_sft_audit_detects_duplicates_without_rewriting_source(tmp_path):
    path = tmp_path / "historical.jsonl"
    original = (
        json.dumps({"id": "a", "user": "Prompt", "assistant": "Answer"})
        + "\n"
        + json.dumps({"id": "b", "user": "Prompt", "assistant": "Answer"})
        + "\n"
    )
    path.write_text(original, encoding="utf-8")

    report = audit_sft_file(path, prompts=())

    assert report["counts"]["raw_duplicates"] == 1
    assert path.read_text(encoding="utf-8") == original
    assert len(report["digest"]) == 64
