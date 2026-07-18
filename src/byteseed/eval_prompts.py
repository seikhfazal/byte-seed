"""Versioned evaluation suites shared by runners and data-quality checks."""

from __future__ import annotations

from dataclasses import dataclass


EVALUATION_PROMPT_REGISTRY_VERSION = 1
RUBRIC_VERSION = 1


@dataclass(frozen=True)
class RubricRequirement:
    """A named concept satisfied by any one normalized phrase."""

    concept: str
    accepted_phrases: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationRubric:
    """A small, transparent deterministic response rubric."""

    version: int
    required: tuple[RubricRequirement, ...]
    forbidden_terms: tuple[str, ...] = ()
    forbidden_combinations: tuple[tuple[str, ...], ...] = ()
    human_review: bool = False


@dataclass(frozen=True)
class EvaluationPrompt:
    prompt_id: str
    text: str
    suite: str
    historical_status: str
    concept_id: str = ""
    rubric: EvaluationRubric | None = None


@dataclass(frozen=True)
class EvaluationSuite:
    suite_id: str
    version: int
    purpose: str
    expected_mode: str
    historical_status: str
    cases: tuple[EvaluationPrompt, ...]


ANCHOR_RETENTION_SUITE = "anchor-retention-v0.2"
ANCHOR_RETENTION_SUITE_VERSION = 1
CANDIDATE_PARAPHRASE_SUITE = "candidate-paraphrase-v1"
CANDIDATE_PARAPHRASE_SUITE_VERSION = 1


def _requirement(concept: str, *phrases: str) -> RubricRequirement:
    return RubricRequirement(concept=concept, accepted_phrases=tuple(phrases))


def _rubric(
    *required: RubricRequirement,
    forbidden_terms: tuple[str, ...] = (),
    forbidden_combinations: tuple[tuple[str, ...], ...] = (),
    human_review: bool = False,
) -> EvaluationRubric:
    return EvaluationRubric(
        version=RUBRIC_VERSION,
        required=tuple(required),
        forbidden_terms=forbidden_terms,
        forbidden_combinations=forbidden_combinations,
        human_review=human_review,
    )


ANCHOR_RETENTION_PROMPTS: tuple[EvaluationPrompt, ...] = (
    EvaluationPrompt(
        "anchor.identity", "who are you?", ANCHOR_RETENTION_SUITE,
        "known-training-overlap", "identity",
        _rubric(_requirement("ByteSeed identity", "byteseed")),
    ),
    EvaluationPrompt(
        "anchor.stack", "what is a stack ?", ANCHOR_RETENTION_SUITE,
        "known-training-overlap", "stack-ordering",
        _rubric(
            _requirement("last-in-first-out ordering", "lifo"),
            forbidden_terms=("bfs",),
        ),
    ),
    EvaluationPrompt(
        "anchor.queue", "What is a queue?", ANCHOR_RETENTION_SUITE,
        "known-training-overlap", "queue-ordering",
        _rubric(
            _requirement("first-in-first-out ordering", "fifo"),
            forbidden_combinations=(("lifo", "stack"),),
        ),
    ),
    EvaluationPrompt(
        "anchor.overfitting", "What is overfitting?", ANCHOR_RETENTION_SUITE,
        "known-training-overlap", "overfitting",
        _rubric(
            _requirement("training data", "training"),
            _requirement("unseen or validation data", "validation", "unseen", "new data"),
        ),
    ),
    EvaluationPrompt(
        "anchor.underfitting", "What is underfitting?", ANCHOR_RETENTION_SUITE,
        "known-training-overlap", "underfitting",
        _rubric(
            _requirement(
                "insufficient model fit", "too simple", "not trained enough",
                "not learn", "too limited",
            ),
            _requirement("poor performance", "poor", "bad", "badly", "high loss"),
            _requirement("training data", "training", "train"),
            _requirement("validation data", "validation"),
        ),
    ),
    EvaluationPrompt(
        "anchor.dsa-plan", "Help me plan a 1 hour DSA study session.",
        ANCHOR_RETENTION_SUITE, "known-training-overlap", "dsa-study-plan",
        _rubric(_requirement("minute-based schedule", "minute")),
    ),
    EvaluationPrompt(
        "anchor.chat-command", "How do I run ByteSeed chat?", ANCHOR_RETENTION_SUITE,
        "known-training-overlap", "chat-command",
        _rubric(
            _requirement("chat launch command", "python chat.py"),
            forbidden_terms=("chat.py.py",),
        ),
    ),
    EvaluationPrompt(
        "anchor.cuda-false", "My PyTorch says CUDA is false. What should I check?",
        ANCHOR_RETENTION_SUITE, "known-training-overlap", "cuda-troubleshooting",
        _rubric(
            _requirement("CUDA", "cuda"),
            _requirement("PyTorch or NVIDIA", "pytorch", "torch", "nvidia"),
        ),
    ),
    EvaluationPrompt(
        "anchor.checkpoint-hygiene", "Should I upload checkpoints to GitHub?",
        ANCHOR_RETENTION_SUITE, "known-training-overlap", "checkpoint-hygiene",
        _rubric(
            _requirement("checkpoint", "checkpoint"),
            _requirement("do not commit checkpoints", "do not commit", "avoid committing"),
        ),
    ),
)


CANDIDATE_PARAPHRASE_PROMPTS: tuple[EvaluationPrompt, ...] = (
    EvaluationPrompt(
        "candidate.identity-introduction", "Introduce yourself in one short sentence.",
        CANDIDATE_PARAPHRASE_SUITE, "candidate-unverified", "identity",
        _rubric(_requirement("ByteSeed identity", "byteseed")),
    ),
    EvaluationPrompt(
        "candidate.stack-ordering", "Which ordering rule does a stack follow?",
        CANDIDATE_PARAPHRASE_SUITE, "candidate-unverified", "stack-ordering",
        _rubric(
            _requirement("last-in-first-out ordering", "lifo", "last in, first out"),
            forbidden_terms=("bfs",),
        ),
    ),
    EvaluationPrompt(
        "candidate.queue-ordering", "How does a queue decide which item leaves next?",
        CANDIDATE_PARAPHRASE_SUITE, "candidate-unverified", "queue-ordering",
        _rubric(
            _requirement("first-in-first-out ordering", "fifo", "first in, first out"),
            forbidden_combinations=(("lifo", "stack"),),
        ),
    ),
    EvaluationPrompt(
        "candidate.overfitting-generalization",
        "Why can a model perform well on its training examples but poorly on unfamiliar data?",
        CANDIDATE_PARAPHRASE_SUITE, "candidate-unverified", "overfitting",
        _rubric(
            _requirement("training data", "training", "train"),
            _requirement("unseen or validation data", "unfamiliar", "unseen", "validation"),
        ),
    ),
    EvaluationPrompt(
        "candidate.underfitting-capacity",
        (
            "What do we call it when a model is too limited to learn patterns in both "
            "training and validation data?"
        ),
        CANDIDATE_PARAPHRASE_SUITE, "candidate-unverified", "underfitting",
        _rubric(
            _requirement("underfitting", "underfitting", "underfit"),
            _requirement("insufficient model fit", "too simple", "too limited", "not learn"),
        ),
    ),
    EvaluationPrompt(
        "candidate.dsa-schedule",
        "Make a sixty-minute practice schedule for data structures and algorithms.",
        CANDIDATE_PARAPHRASE_SUITE, "candidate-unverified", "dsa-study-plan",
        _rubric(
            _requirement("minute-based schedule", "minute", "minutes", "60", "sixty"),
            human_review=True,
        ),
    ),
    EvaluationPrompt(
        "candidate.local-chat-launch",
        "Which command starts the local ByteSeed terminal chat?",
        CANDIDATE_PARAPHRASE_SUITE, "candidate-unverified", "chat-command",
        _rubric(
            _requirement("chat launch command", "python chat.py"),
            forbidden_terms=("chat.py.py",),
        ),
    ),
    EvaluationPrompt(
        "candidate.cuda-detection",
        "What should I inspect when Torch cannot see my NVIDIA GPU?",
        CANDIDATE_PARAPHRASE_SUITE, "candidate-unverified", "cuda-troubleshooting",
        _rubric(
            _requirement("CUDA", "cuda"),
            _requirement("PyTorch or NVIDIA", "pytorch", "torch", "nvidia"),
        ),
    ),
    EvaluationPrompt(
        "candidate.checkpoint-git-policy",
        "What is the safe Git policy for locally generated model checkpoint files?",
        CANDIDATE_PARAPHRASE_SUITE, "candidate-unverified", "checkpoint-hygiene",
        _rubric(
            _requirement("checkpoint", "checkpoint"),
            _requirement(
                "do not commit checkpoints", "do not commit", "avoid committing",
                "keep them out", "gitignore", ".gitignore",
            ),
        ),
    ),
)


ANCHOR_RETENTION_DEFINITION = EvaluationSuite(
    ANCHOR_RETENTION_SUITE,
    ANCHOR_RETENTION_SUITE_VERSION,
    "retention",
    "deterministic-rubric",
    "known-contaminated",
    ANCHOR_RETENTION_PROMPTS,
)
CANDIDATE_PARAPHRASE_DEFINITION = EvaluationSuite(
    CANDIDATE_PARAPHRASE_SUITE,
    CANDIDATE_PARAPHRASE_SUITE_VERSION,
    "candidate-generalization",
    "deterministic-rubric",
    "candidate-unverified",
    CANDIDATE_PARAPHRASE_PROMPTS,
)


def validate_evaluation_suite(suite: EvaluationSuite) -> None:
    if not suite.suite_id.strip():
        raise ValueError("evaluation suite ID must be non-empty")
    if suite.version != 1:
        raise ValueError(f"unsupported evaluation suite version: {suite.version}")
    if not suite.cases:
        raise ValueError(f"evaluation suite {suite.suite_id!r} has no cases")

    prompt_ids: set[str] = set()
    for case in suite.cases:
        if case.suite != suite.suite_id:
            raise ValueError(
                f"evaluation case {case.prompt_id!r} belongs to {case.suite!r}, "
                f"not {suite.suite_id!r}"
            )
        if not case.prompt_id.strip() or not case.text.strip():
            raise ValueError("evaluation prompt IDs and text must be non-empty")
        if case.prompt_id in prompt_ids:
            raise ValueError(f"duplicate evaluation prompt ID: {case.prompt_id!r}")
        prompt_ids.add(case.prompt_id)


def registered_evaluation_suites() -> tuple[EvaluationSuite, ...]:
    suites = (ANCHOR_RETENTION_DEFINITION, CANDIDATE_PARAPHRASE_DEFINITION)
    for suite in suites:
        validate_evaluation_suite(suite)
    return suites


def get_evaluation_suite(suite_id: str) -> EvaluationSuite:
    for suite in registered_evaluation_suites():
        if suite.suite_id == suite_id:
            return suite
    available = ", ".join(suite.suite_id for suite in registered_evaluation_suites())
    raise ValueError(f"unknown evaluation suite {suite_id!r}; choose one of: {available}")


def serialize_evaluation_suite(suite: EvaluationSuite) -> dict[str, object]:
    """Return the suite's deterministic, JSON-compatible identity payload."""

    validate_evaluation_suite(suite)
    cases: list[dict[str, object]] = []
    for case in suite.cases:
        rubric = case.rubric
        rubric_payload: dict[str, object] | None = None
        if rubric is not None:
            rubric_payload = {
                "version": rubric.version,
                "required": [
                    {
                        "concept": requirement.concept,
                        "accepted_phrases": list(requirement.accepted_phrases),
                    }
                    for requirement in rubric.required
                ],
                "forbidden_terms": list(rubric.forbidden_terms),
                "forbidden_combinations": [
                    list(combination) for combination in rubric.forbidden_combinations
                ],
                "human_review": rubric.human_review,
            }
        cases.append(
            {
                "prompt_id": case.prompt_id,
                "text": case.text,
                "concept_id": case.concept_id,
                "historical_status": case.historical_status,
                "rubric": rubric_payload,
            }
        )
    return {
        "suite_id": suite.suite_id,
        "version": suite.version,
        "purpose": suite.purpose,
        "expected_mode": suite.expected_mode,
        "historical_status": suite.historical_status,
        "cases": cases,
    }


def registered_evaluation_prompts() -> tuple[EvaluationPrompt, ...]:
    """Return prompts in stable suite and case order."""

    return tuple(case for suite in registered_evaluation_suites() for case in suite.cases)
