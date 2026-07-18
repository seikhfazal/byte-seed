from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationPrompt:
    """A stable input used by a named ByteSeed evaluation suite."""

    prompt_id: str
    text: str
    suite: str
    historical_status: str


ANCHOR_RETENTION_SUITE = "anchor-retention-v0.2"

# Keep this tuple in the same order as the historical stable-v0.2 script. These
# prompts occur verbatim in Anchor v2.3 SFT data and are retention checks, not a
# held-out generalization suite.
ANCHOR_RETENTION_PROMPTS = (
    EvaluationPrompt("anchor.identity", "who are you?", ANCHOR_RETENTION_SUITE, "known-training-overlap"),
    EvaluationPrompt("anchor.stack", "what is a stack ?", ANCHOR_RETENTION_SUITE, "known-training-overlap"),
    EvaluationPrompt("anchor.queue", "What is a queue?", ANCHOR_RETENTION_SUITE, "known-training-overlap"),
    EvaluationPrompt("anchor.overfitting", "What is overfitting?", ANCHOR_RETENTION_SUITE, "known-training-overlap"),
    EvaluationPrompt("anchor.underfitting", "What is underfitting?", ANCHOR_RETENTION_SUITE, "known-training-overlap"),
    EvaluationPrompt(
        "anchor.dsa-plan",
        "Help me plan a 1 hour DSA study session.",
        ANCHOR_RETENTION_SUITE,
        "known-training-overlap",
    ),
    EvaluationPrompt("anchor.chat-command", "How do I run ByteSeed chat?", ANCHOR_RETENTION_SUITE, "known-training-overlap"),
    EvaluationPrompt(
        "anchor.cuda-false",
        "My PyTorch says CUDA is false. What should I check?",
        ANCHOR_RETENTION_SUITE,
        "known-training-overlap",
    ),
    EvaluationPrompt(
        "anchor.checkpoint-hygiene",
        "Should I upload checkpoints to GitHub?",
        ANCHOR_RETENTION_SUITE,
        "known-training-overlap",
    ),
)


def registered_evaluation_prompts() -> tuple[EvaluationPrompt, ...]:
    """Return registered prompts in stable suite order."""

    return ANCHOR_RETENTION_PROMPTS
