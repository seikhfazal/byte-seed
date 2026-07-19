from __future__ import annotations

import torch

from byteseed.chat import clean_assistant_output, is_degenerate_reply, print_banner


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


def test_chat_banner_identifies_resolved_attention_backend(capsys):
    print_banner(
        "ByteSeed-Test",
        17,
        torch.device("cpu"),
        "synthetic.pt",
        "precise",
        0.2,
        5,
        80,
        False,
        1.0,
        "fp32",
        False,
        "sdpa",
    )

    assert "Attention backend: sdpa" in capsys.readouterr().out
