from __future__ import annotations

from byteseed.chat import clean_assistant_output, is_degenerate_reply


def test_cleanup_trims_special_tokens_and_artificial_labels():
    assert clean_assistant_output("Answer<|end|><|end|> trailing") == "Answer"
    assert clean_assistant_output("Keep checkpoints local. Command note 7.") == "Keep checkpoints local."


def test_cleanup_does_not_rewrite_normal_answers():
    answer = "Use a stack contrast in your explanation, but keep it brief."

    assert clean_assistant_output(answer) == answer


def test_empty_and_punctuation_only_replies_are_degenerate():
    assert is_degenerate_reply("   ")
    assert is_degenerate_reply("?!...")
    assert not is_degenerate_reply("A useful answer.")
