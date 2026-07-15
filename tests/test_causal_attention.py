from __future__ import annotations

import torch


def test_attention_output_shape_for_short_sequence(tiny_config, tiny_model):
    activations = torch.randn(2, 3, tiny_config.n_embd)

    output = tiny_model.blocks[0].attn(activations)

    assert output.shape == activations.shape


def test_future_token_cannot_change_earlier_logits(tiny_model):
    original = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    changed_future = original.clone()
    changed_future[0, -1] = 6

    original_logits, _ = tiny_model(original)
    changed_logits, _ = tiny_model(changed_future)

    assert torch.allclose(original_logits[:, :-1], changed_logits[:, :-1], atol=0.0, rtol=0.0)


def test_attention_handles_sequence_exactly_at_block_size(tiny_config, tiny_model):
    tokens = torch.randint(tiny_config.vocab_size, (1, tiny_config.block_size), dtype=torch.long)

    logits, _ = tiny_model(tokens)

    assert logits.shape[:2] == tokens.shape
