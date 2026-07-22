from __future__ import annotations

import copy

import pytest

from byteseed.eval_prompts import (
    ANCHOR_RETENTION_DEFINITION,
    ANCHOR_RETENTION_PROMPTS,
    ANCHOR_RETENTION_SUITE,
    ANCHOR_RETENTION_SUITE_VERSION,
    CANDIDATE_PARAPHRASE_DEFINITION,
    CANDIDATE_PARAPHRASE_PROMPTS,
    CANDIDATE_PARAPHRASE_SUITE,
    CANDIDATE_PARAPHRASE_SUITE_VERSION,
    EvaluationSuite,
    GENERALIZATION_HOLDOUT_DEFINITION,
    GENERALIZATION_HOLDOUT_PROMPTS,
    GENERALIZATION_HOLDOUT_SUITE,
    GENERALIZATION_HOLDOUT_SUITE_VERSION,
    registered_evaluation_prompts,
    serialize_evaluation_suite,
    validate_evaluation_suite,
)
from byteseed.evaluation import (
    GeneratedCaseOutput,
    GenerationConfig,
    normalize_rubric_text,
    run_evaluation,
    score_response,
)
from byteseed.provenance import canonical_sha256
from scripts.eval_stable_v0_2 import check_answer, main as evaluation_main


ANCHOR_TEXT = [
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


def test_anchor_suite_identity_text_order_and_purpose_are_stable():
    assert ANCHOR_RETENTION_SUITE == "anchor-retention-v0.2"
    assert ANCHOR_RETENTION_SUITE_VERSION == 1
    assert ANCHOR_RETENTION_DEFINITION.purpose == "retention"
    assert ANCHOR_RETENTION_DEFINITION.historical_status == "known-contaminated"
    assert [case.text for case in ANCHOR_RETENTION_PROMPTS] == ANCHOR_TEXT
    assert len({case.prompt_id for case in ANCHOR_RETENTION_PROMPTS}) == 9


def test_candidate_suite_is_versioned_distinct_and_initially_unverified():
    assert CANDIDATE_PARAPHRASE_SUITE == "candidate-paraphrase-v1"
    assert CANDIDATE_PARAPHRASE_SUITE_VERSION == 1
    assert CANDIDATE_PARAPHRASE_DEFINITION.purpose == "candidate-generalization"
    assert CANDIDATE_PARAPHRASE_DEFINITION.historical_status == "candidate-unverified"
    assert len(CANDIDATE_PARAPHRASE_PROMPTS) == len(ANCHOR_RETENTION_PROMPTS) == 9
    assert len({case.prompt_id for case in CANDIDATE_PARAPHRASE_PROMPTS}) == 9
    assert {
        normalize_rubric_text(case.text) for case in CANDIDATE_PARAPHRASE_PROMPTS
    }.isdisjoint({normalize_rubric_text(case.text) for case in ANCHOR_RETENTION_PROMPTS})


def test_candidate_has_one_case_per_anchor_concept_and_transparent_rubrics():
    assert [case.concept_id for case in CANDIDATE_PARAPHRASE_PROMPTS] == [
        case.concept_id for case in ANCHOR_RETENTION_PROMPTS
    ]
    assert all(case.rubric is not None for case in CANDIDATE_PARAPHRASE_PROMPTS)
    for case in CANDIDATE_PARAPHRASE_PROMPTS:
        assert case.rubric is not None
        assert case.rubric.version == 1
        assert case.rubric.required
        for requirement in case.rubric.required:
            assert requirement.concept
            normalized = tuple(
                normalize_rubric_text(phrase)
                for phrase in requirement.accepted_phrases
            )
            assert tuple(normalize_rubric_text(phrase) for phrase in normalized) == normalized
            assert all(normalized)


def test_registry_order_and_suite_serialization_are_deterministic():
    assert registered_evaluation_prompts() == (
        ANCHOR_RETENTION_PROMPTS
        + CANDIDATE_PARAPHRASE_PROMPTS
        + GENERALIZATION_HOLDOUT_PROMPTS
    )
    first = serialize_evaluation_suite(CANDIDATE_PARAPHRASE_DEFINITION)
    second = copy.deepcopy(first)
    assert first == second
    assert canonical_sha256(first) == canonical_sha256(second)
    assert [case["prompt_id"] for case in first["cases"]] == [
        case.prompt_id for case in CANDIDATE_PARAPHRASE_PROMPTS
    ]


def test_existing_suite_payloads_are_frozen_by_canonical_digest():
    assert canonical_sha256(serialize_evaluation_suite(ANCHOR_RETENTION_DEFINITION)) == (
        "87aaf506deb5c4085403314cdc87e3548eb5038210b6957170b2356a8af59ff3"
    )
    assert canonical_sha256(serialize_evaluation_suite(CANDIDATE_PARAPHRASE_DEFINITION)) == (
        "8e00b0f4f53603bd57f7f734f5a8fb95fc43f1201112a2181398cca60f2e5b20"
    )
    assert [case.prompt_id for case in ANCHOR_RETENTION_PROMPTS] == [
        "anchor.identity",
        "anchor.stack",
        "anchor.queue",
        "anchor.overfitting",
        "anchor.underfitting",
        "anchor.dsa-plan",
        "anchor.chat-command",
        "anchor.cuda-false",
        "anchor.checkpoint-hygiene",
    ]
    assert [case.prompt_id for case in CANDIDATE_PARAPHRASE_PROMPTS] == [
        "candidate.identity-introduction",
        "candidate.stack-ordering",
        "candidate.queue-ordering",
        "candidate.overfitting-generalization",
        "candidate.underfitting-capacity",
        "candidate.dsa-schedule",
        "candidate.local-chat-launch",
        "candidate.cuda-detection",
        "candidate.checkpoint-git-policy",
    ]


def test_generalization_holdout_is_versioned_balanced_and_unverified():
    assert GENERALIZATION_HOLDOUT_SUITE == "generalization-holdout-v1"
    assert GENERALIZATION_HOLDOUT_SUITE_VERSION == 1
    assert GENERALIZATION_HOLDOUT_DEFINITION.purpose == "candidate-generalization"
    assert GENERALIZATION_HOLDOUT_DEFINITION.historical_status == "candidate-unverified"
    assert len(GENERALIZATION_HOLDOUT_PROMPTS) == 24
    assert len({case.prompt_id for case in GENERALIZATION_HOLDOUT_PROMPTS}) == 24
    concepts = {}
    for case in GENERALIZATION_HOLDOUT_PROMPTS:
        concepts[case.concept_id] = concepts.get(case.concept_id, 0) + 1
        assert case.rubric is not None
        assert case.rubric.required
    assert concepts == {
        "identity": 2,
        "capabilities-limitations": 2,
        "stack-fundamentals": 2,
        "queue-fundamentals": 2,
        "stack-queue-comparison": 2,
        "overfitting": 2,
        "underfitting": 2,
        "fit-contrast": 2,
        "dsa-study-planning": 2,
        "local-workflow": 2,
        "cuda-troubleshooting": 2,
        "checkpoint-git-hygiene": 2,
    }
    assert {
        normalize_rubric_text(case.text) for case in GENERALIZATION_HOLDOUT_PROMPTS
    }.isdisjoint(
        {
            normalize_rubric_text(case.text)
            for case in ANCHOR_RETENTION_PROMPTS + CANDIDATE_PARAPHRASE_PROMPTS
        }
    )


def test_generalization_holdout_report_is_deterministic_and_not_claimed_held_out():
    def generator(cases, _config):
        return [
            GeneratedCaseOutput("unscored synthetic answer", 3, "max_new_tokens")
            for _ in cases
        ]

    environment = {
        "python_version": "test",
        "pytorch_version": "test",
        "device": "cpu",
        "dtype": "fp32",
        "compile": False,
        "deterministic_algorithms_enabled": False,
    }
    first = run_evaluation(
        GENERALIZATION_HOLDOUT_DEFINITION,
        GenerationConfig(seed=19),
        generator,
        environment=environment,
    )
    second = run_evaluation(
        GENERALIZATION_HOLDOUT_DEFINITION,
        GenerationConfig(seed=19),
        generator,
        environment=environment,
    )
    assert first["digest"] == second["digest"]
    assert first["summary"]["total_cases"] == 24
    assert first["summary"]["metric_label"] == "Generalization holdout candidate checks"
    assert first["summary"]["held_out_generalization_measured"] is False
    assert first["contamination"]["held_out_status"] == "unverified"


def test_holdout_fit_contradiction_rubrics_reject_concept_blending():
    underfit = next(
        case
        for case in GENERALIZATION_HOLDOUT_PROMPTS
        if case.prompt_id == "generalization.underfitting.both-splits"
    )
    assert score_response(
        underfit,
        "This is underfitting because training and validation are both weak.",
    )["status"] == "pass"
    assert score_response(
        underfit,
        "This is overfitting even though training and validation are both weak.",
    )["status"] == "fail"


def test_stable_evaluation_cli_lists_the_new_suite(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["eval_stable_v0_2.py", "--help"])
    with pytest.raises(SystemExit) as error:
        evaluation_main()
    assert error.value.code == 0
    assert "generalization-holdout-v1" in capsys.readouterr().out


def test_duplicate_prompt_ids_fail_clearly():
    case = ANCHOR_RETENTION_PROMPTS[0]
    duplicate = EvaluationSuite(
        suite_id=case.suite,
        version=1,
        purpose="retention",
        expected_mode="deterministic-rubric",
        historical_status="known-contaminated",
        cases=(case, case),
    )
    with pytest.raises(ValueError, match="duplicate evaluation prompt ID"):
        validate_evaluation_suite(duplicate)


def test_required_concepts_pass_and_missing_concepts_fail_transparently():
    stack = CANDIDATE_PARAPHRASE_PROMPTS[1]
    passed = score_response(stack, "A stack follows LIFO: last in, first out.")
    failed = score_response(stack, "It stores several items.")

    assert passed["status"] == "pass"
    assert passed["passed"] is True
    assert failed["status"] == "fail"
    assert failed["passed"] is False
    assert failed["missing_required_concepts"] == ["last-in-first-out ordering"]


def test_forbidden_concepts_are_only_applied_when_declared():
    stack = score_response(
        CANDIDATE_PARAPHRASE_PROMPTS[1],
        "A stack uses LIFO, not BFS.",
    )
    queue = score_response(
        CANDIDATE_PARAPHRASE_PROMPTS[2],
        "A queue is FIFO; a stack is LIFO.",
    )

    assert stack["status"] == "fail"
    assert stack["forbidden_matches"] == ["bfs"]
    assert queue["status"] == "fail"
    assert queue["forbidden_matches"] == ["lifo + stack"]


def test_human_review_case_is_visible_and_not_silently_passing():
    score = score_response(
        CANDIDATE_PARAPHRASE_PROMPTS[5],
        "Use 20 minutes for review and 40 minutes for practice.",
    )
    assert score["status"] == "human-review"
    assert score["passed"] is None
    assert score["human_review_required"] is True


@pytest.mark.parametrize(
    ("prompt_index", "answer"),
    [
        (0, "I am ByteSeed."),
        (0, ""),
        (1, "A stack is LIFO."),
        (1, "A stack is LIFO and BFS."),
        (2, "A queue is FIFO."),
        (2, "A queue is FIFO while a stack is LIFO."),
        (3, "It fits training data but fails on unseen data."),
        (4, "The model is too simple and has poor training and validation results."),
        (4, "The model is too simple."),
        (5, "Use 20 minutes to review and 40 minutes to practice."),
        (6, "Run python chat.py."),
        (6, "Run python chat.py.py."),
        (7, "Check the PyTorch CUDA build and NVIDIA driver."),
        (8, "Avoid committing checkpoint files."),
    ],
)
def test_shared_anchor_rubrics_preserve_stable_script_pass_fail_semantics(
    prompt_index,
    answer,
):
    case = ANCHOR_RETENTION_PROMPTS[prompt_index]
    legacy_passed, _reason = check_answer(case.text, answer)
    assert score_response(case, answer)["passed"] is legacy_passed
