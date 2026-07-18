from __future__ import annotations

import json
from pathlib import Path

import pytest

from byteseed.data_quality import (
    EvaluationContaminationError,
    audit_sft_file,
    build_data_quality_report,
    create_document,
    detect_evaluation_contamination,
    plan_document_dataset,
    read_sft_documents,
)
from byteseed.eval_prompts import (
    ANCHOR_RETENTION_PROMPTS,
    CANDIDATE_PARAPHRASE_PROMPTS,
    EvaluationPrompt,
    registered_evaluation_prompts,
)


ROOT = Path(__file__).resolve().parents[1]


def _prompt(text: str = "What is a queue?") -> EvaluationPrompt:
    return EvaluationPrompt("test.queue", text, "test-suite", "synthetic")


def _clean_document(index: int):
    return create_document(
        text=f"Unrelated document {index}",
        source=f"fixture/{index}.md",
        document_id=f"clean-{index}",
    )


def test_full_document_contamination_is_detected():
    document = create_document(
        text="What is a queue?", source="fixture.md", document_id="full"
    )

    findings = detect_evaluation_contamination([document], prompts=[_prompt()])

    assert len(findings) == 1
    assert findings[0].match_type == "full-document"


def test_structured_user_field_exact_match_is_detected():
    document = create_document(
        text="<|user|>\nWhat is a queue?\n<|assistant|>\nA FIFO structure.\n<|end|>",
        source="sft.jsonl",
        document_id="field",
        fields={"user": "What is a queue?", "assistant": "A FIFO structure."},
    )

    findings = detect_evaluation_contamination([document], prompts=[_prompt()])

    assert findings[0].match_type == "field-exact"


def test_normalized_substring_and_whitespace_variants_are_detected():
    document = create_document(
        text="Prefix\r\nWhat   is a queue?\r\nSuffix",
        source="notes.md",
        document_id="substring",
    )

    findings = detect_evaluation_contamination([document], prompts=[_prompt()])

    assert findings[0].match_type == "normalized-substring"


def test_unrelated_text_is_not_flagged():
    assert detect_evaluation_contamination(
        [_clean_document(1)], prompts=[_prompt()]
    ) == ()


def test_equivalent_field_and_combined_matches_are_deduplicated():
    document = create_document(
        text="Question: What is a queue?",
        source="sft.jsonl",
        document_id="one-logical-match",
        fields={"user": "What is a queue?"},
    )

    findings = detect_evaluation_contamination([document], prompts=[_prompt()])

    assert len(findings) == 1
    assert findings[0].match_type == "field-exact"


def test_finding_contains_ids_and_short_hash_but_not_private_text():
    document = create_document(
        text="Private surrounding text. What is a queue? More private text.",
        source="private.md",
        document_id="private-doc",
    )

    finding = detect_evaluation_contamination([document], prompts=[_prompt()])[0]
    payload = finding.as_dict()

    assert payload["prompt_id"] == "test.queue"
    assert payload["document_id"] == "private-doc"
    assert payload["source"] == "private.md"
    assert len(payload["document_fingerprint"]) == 12
    assert "Private surrounding text" not in json.dumps(payload)


def test_default_policy_rejects_registered_contamination():
    contaminated = create_document(
        text="What is a queue?", source="fixture.md", document_id="bad"
    )

    with pytest.raises(EvaluationContaminationError, match="allow-eval-contamination"):
        plan_document_dataset(
            [contaminated, _clean_document(2)],
            seed=1,
            validation_ratio=0.5,
            prompts=[_prompt()],
        )


def test_explicit_override_warns_and_is_recorded_as_not_held_out():
    contaminated = create_document(
        text="What is a queue?", source="fixture.md", document_id="accepted-history"
    )

    with pytest.warns(UserWarning, match="not held-out generalization"):
        plan = plan_document_dataset(
            [contaminated, _clean_document(2)],
            seed=1,
            validation_ratio=0.5,
            prompts=[_prompt()],
            allow_eval_contamination=True,
        )
    report = build_data_quality_report(plan)

    assert report["counts"]["contamination_matches"] == 1
    assert report["policy"]["allow_eval_contamination"] is True
    assert report["policy"]["held_out_generalization_measured"] is False
    assert report["contamination_findings"][0]["dataset_split"] in {
        "train",
        "validation",
    }


def test_anchor_prompt_registry_is_unique_deterministic_and_text_stable():
    expected = [
        "who are you?",
        "what is a stack ?",
        "What is a queue?",
        "What is overfitting?",
        "What is underfitting?",
        "Help me plan a 1 hour DSA study session.",
        "How do I run ByteSeed chat?",
        "My PyTorch says CUDA is false. What should I check?",
        "Should I upload checkpoints to GitHub?",
    ]

    registry = registered_evaluation_prompts()

    assert registry == ANCHOR_RETENTION_PROMPTS + CANDIDATE_PARAPHRASE_PROMPTS
    assert [prompt.text for prompt in ANCHOR_RETENTION_PROMPTS] == expected
    assert len({prompt.prompt_id for prompt in registry}) == len(registry)
    assert all(
        prompt.historical_status == "known-training-overlap"
        for prompt in ANCHOR_RETENTION_PROMPTS
    )


def test_stable_evaluation_reports_retention_separately_from_generalization():
    wrapper = (ROOT / "scripts" / "eval_stable_v0_2.py").read_text(encoding="utf-8")
    renderer = (ROOT / "src" / "byteseed" / "evaluation.py").read_text(
        encoding="utf-8"
    )

    assert "render_evaluation_report" in wrapper
    assert "Anchor-retention regression" in renderer
    assert "Held-out generalization: not yet measured." in renderer


def test_historical_anchor_v2_3_sft_contains_all_retention_prompts():
    documents = read_sft_documents(
        ROOT / "examples" / "byteseed_anchor_v2_3_sft.jsonl"
    )

    findings = detect_evaluation_contamination(documents)

    assert {finding.prompt_id for finding in findings} == {
        prompt.prompt_id for prompt in ANCHOR_RETENTION_PROMPTS
    }
    assert all(finding.match_type == "field-exact" for finding in findings)


def test_historical_sft_audit_reports_overlap_without_rewriting(tmp_path):
    path = tmp_path / "history.jsonl"
    original = json.dumps(
        {"user": "What is a queue?", "assistant": "A queue is FIFO."}
    ) + "\n"
    path.write_text(original, encoding="utf-8")

    report = audit_sft_file(path, prompts=[_prompt()])

    assert report["counts"]["contamination_matches"] == 1
    assert report["held_out_generalization_measured"] is False
    assert path.read_text(encoding="utf-8") == original
