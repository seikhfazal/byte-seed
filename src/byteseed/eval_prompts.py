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
GENERALIZATION_HOLDOUT_SUITE = "generalization-holdout-v1"
GENERALIZATION_HOLDOUT_SUITE_VERSION = 1


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


GENERALIZATION_HOLDOUT_PROMPTS: tuple[EvaluationPrompt, ...] = (
    EvaluationPrompt(
        "generalization.identity.local-model",
        "A user mistakes this project for a hosted service. In one sentence, what is ByteSeed?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "identity",
        _rubric(
            _requirement("ByteSeed identity", "byteseed"),
            _requirement("small local model", "small", "tiny", "local"),
        ),
    ),
    EvaluationPrompt(
        "generalization.identity.learning-role",
        "Describe ByteSeed's intended role without presenting it as production software.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "identity",
        _rubric(
            _requirement("ByteSeed identity", "byteseed"),
            _requirement("learning or experimentation", "learning", "experiment"),
        ),
    ),
    EvaluationPrompt(
        "generalization.capabilities.modest-tasks",
        "Give two modest kinds of technical help this small project is designed to demonstrate.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "capabilities-limitations",
        _rubric(
            _requirement("technical learning task", "dsa", "data structure", "study"),
            _requirement("local project help", "byteseed", "pytorch", "cuda", "local"),
        ),
    ),
    EvaluationPrompt(
        "generalization.capabilities.boundary",
        "What important limit should accompany a truthful summary of ByteSeed's capabilities?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "capabilities-limitations",
        _rubric(
            _requirement(
                "limited experimental scope", "small", "limited", "experimental",
                "not production", "cannot",
            ),
            forbidden_terms=("guaranteed correctness", "production ready"),
        ),
    ),
    EvaluationPrompt(
        "generalization.stack.plate-removal",
        "Four plates are placed down in order A, B, C, then D. Using stack behavior, which leaves first?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "stack-fundamentals",
        _rubric(
            _requirement("D leaves first", "plate d", "item d", "d leaves", "d is first"),
            _requirement("last-in-first-out ordering", "lifo", "last in", "most recent", "top"),
            forbidden_terms=("fifo",),
        ),
    ),
    EvaluationPrompt(
        "generalization.stack.undo-history",
        "Why is a stack a natural structure for an editor's undo history?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "stack-fundamentals",
        _rubric(
            _requirement("most recent action first", "most recent", "latest", "last action"),
            _requirement("stack operation or order", "pop", "lifo", "last in"),
            forbidden_terms=("fifo",),
        ),
    ),
    EvaluationPrompt(
        "generalization.queue.printer-order",
        "Print jobs R, S, and T arrive in that order. Which job should a normal queue process first, and why?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "queue-fundamentals",
        _rubric(
            _requirement("R leaves first", "job r", "request r", "r leaves", "r is first"),
            _requirement("first-in-first-out ordering", "fifo", "first in", "arrived first"),
            forbidden_terms=("lifo",),
        ),
    ),
    EvaluationPrompt(
        "generalization.queue.service-line",
        "Explain why a fair first-come service line is modeled with a queue.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "queue-fundamentals",
        _rubric(
            _requirement("arrival order", "first come", "arrived first", "oldest"),
            _requirement("queue ordering", "fifo", "first in"),
            forbidden_terms=("lifo",),
        ),
    ),
    EvaluationPrompt(
        "generalization.stack-queue.undo-print",
        "Choose structures for editor undo actions and printer jobs, and state the ordering rule for each.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "stack-queue-comparison",
        _rubric(
            _requirement("undo uses stack", "undo", "stack"),
            _requirement("printer uses queue", "printer", "queue"),
            _requirement("both ordering rules", "lifo", "last in"),
            _requirement("queue ordering", "fifo", "first in"),
        ),
    ),
    EvaluationPrompt(
        "generalization.stack-queue.next-item",
        "How does choosing the next item differ between a stack and a queue?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "stack-queue-comparison",
        _rubric(
            _requirement("stack takes newest", "stack", "lifo", "newest", "last"),
            _requirement("queue takes oldest", "queue", "fifo", "oldest", "first"),
        ),
    ),
    EvaluationPrompt(
        "generalization.overfitting.loss-curves",
        "Training loss keeps falling while validation loss rises. Name the likely problem and explain the signal.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "overfitting",
        _rubric(
            _requirement("overfitting", "overfit"),
            _requirement("training improves", "training loss", "train"),
            _requirement("validation worsens", "validation loss", "validation"),
        ),
    ),
    EvaluationPrompt(
        "generalization.overfitting.memorized-drills",
        "A learner reproduces practice answers exactly but fails new versions of the problems. What ML failure does this resemble?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "overfitting",
        _rubric(
            _requirement("overfitting", "overfit"),
            _requirement("memorization", "memoriz"),
            _requirement("weak generalization", "new", "unseen", "generaliz"),
        ),
    ),
    EvaluationPrompt(
        "generalization.underfitting.both-splits",
        "A model has large errors on both its training set and validation set. Which fit problem is most likely?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "underfitting",
        _rubric(
            _requirement("underfitting", "underfit"),
            _requirement("both splits are weak", "training", "train"),
            _requirement("validation is weak", "validation"),
            forbidden_terms=("overfitting",),
        ),
    ),
    EvaluationPrompt(
        "generalization.underfitting.insufficient-model",
        "What happens when a model is too simple to capture even the main pattern in its examples?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "underfitting",
        _rubric(
            _requirement("underfitting", "underfit"),
            _requirement("insufficient fit", "too simple", "capacity", "cannot capture"),
            forbidden_terms=("memorization",),
        ),
    ),
    EvaluationPrompt(
        "generalization.fit-contrast.two-models",
        "Model A is strong on training data but weak on validation; Model B is weak on both. Classify each.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "fit-contrast",
        _rubric(
            _requirement("A overfits", "model a overfits", "a is overfitting", "a overfits"),
            _requirement("B underfits", "model b underfits", "b is underfitting", "b underfits"),
        ),
    ),
    EvaluationPrompt(
        "generalization.fit-contrast.diagnostic",
        "Use training and validation behavior to distinguish overfitting from underfitting.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "fit-contrast",
        _rubric(
            _requirement("overfit split gap", "overfit", "training", "validation"),
            _requirement("underfit both weak", "underfit", "both", "training and validation"),
        ),
    ),
    EvaluationPrompt(
        "generalization.dsa-plan.forty-five-minutes",
        "Design a focused 45-minute session for a weak data-structure topic.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "dsa-study-planning",
        _rubric(
            _requirement("time allocation", "minute", "minutes"),
            _requirement("practice", "practice", "solve", "problem"),
            _requirement("review", "review", "mistake", "recap"),
        ),
    ),
    EvaluationPrompt(
        "generalization.dsa-plan.twenty-five-minutes",
        "Only 25 minutes remain before a study break. Give a realistic DSA practice plan.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "dsa-study-planning",
        _rubric(
            _requirement("time allocation", "minute", "minutes"),
            _requirement("practice", "practice", "solve", "problem"),
        ),
    ),
    EvaluationPrompt(
        "generalization.local-workflow.venv-ready",
        "The repository is cloned and the virtual environment is active. What command opens its terminal interface?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "local-workflow",
        _rubric(
            _requirement("chat launch command", "python chat.py"),
            forbidden_terms=("chat.py.py",),
        ),
    ),
    EvaluationPrompt(
        "generalization.local-workflow.wrong-directory",
        "A PowerShell session is outside the project folder. Give the two basic steps needed to start ByteSeed locally.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "local-workflow",
        _rubric(
            _requirement("enter repository", "cd", "set-location"),
            _requirement("chat launch command", "python chat.py"),
            forbidden_terms=("chat.py.py",),
        ),
    ),
    EvaluationPrompt(
        "generalization.cuda.runtime-mismatch",
        "nvidia-smi works, but torch.cuda.is_available() is false. Name two targeted checks.",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "cuda-troubleshooting",
        _rubric(
            _requirement("PyTorch CUDA build", "torch.version.cuda", "cuda build", "pytorch build"),
            _requirement("active environment", "environment", "venv", "interpreter"),
        ),
    ),
    EvaluationPrompt(
        "generalization.cuda.cpu-build",
        "PyTorch reports that it was compiled without CUDA support. What is the practical next step?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "cuda-troubleshooting",
        _rubric(
            _requirement("install CUDA-enabled PyTorch", "cuda-enabled", "cuda build", "reinstall"),
            _requirement("matching environment or platform", "environment", "pytorch", "driver", "system"),
        ),
    ),
    EvaluationPrompt(
        "generalization.checkpoint.large-file",
        "A 600 MB .pt file appears in git status after training. What should happen before the next commit?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "checkpoint-git-hygiene",
        _rubric(
            _requirement("exclude checkpoint", "do not commit", "remove", "keep it out"),
            _requirement("ignore rule", ".gitignore", "gitignore"),
        ),
    ),
    EvaluationPrompt(
        "generalization.checkpoint.distribution",
        "How can trained weights be shared without placing generated binaries in ordinary source history?",
        GENERALIZATION_HOLDOUT_SUITE, "candidate-unverified", "checkpoint-git-hygiene",
        _rubric(
            _requirement(
                "external artifact storage", "release asset", "model registry",
                "object storage", "artifact storage",
            ),
            _requirement("source history distinction", "git", "repository", "source"),
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
GENERALIZATION_HOLDOUT_DEFINITION = EvaluationSuite(
    GENERALIZATION_HOLDOUT_SUITE,
    GENERALIZATION_HOLDOUT_SUITE_VERSION,
    "candidate-generalization",
    "deterministic-rubric",
    "candidate-unverified",
    GENERALIZATION_HOLDOUT_PROMPTS,
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
    suites = (
        ANCHOR_RETENTION_DEFINITION,
        CANDIDATE_PARAPHRASE_DEFINITION,
        GENERALIZATION_HOLDOUT_DEFINITION,
    )
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
