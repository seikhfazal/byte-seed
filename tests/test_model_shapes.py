from __future__ import annotations

import pytest
import torch


def test_model_forward_shapes_loss_and_cpu_compatibility(tiny_config, tiny_model):
    idx = torch.randint(tiny_config.vocab_size, (2, 4), dtype=torch.long)
    targets = torch.randint(tiny_config.vocab_size, (2, 4), dtype=torch.long)

    logits, loss = tiny_model(idx, targets)

    assert logits.shape == (2, 4, tiny_config.vocab_size)
    assert loss is not None and loss.ndim == 0
    assert torch.isfinite(loss)
    assert sum(parameter.numel() for parameter in tiny_model.parameters()) > 0
    assert next(tiny_model.parameters()).device.type == "cpu"


def test_model_accepts_exact_block_size_and_rejects_longer_input(tiny_config, tiny_model):
    exact = torch.randint(tiny_config.vocab_size, (1, tiny_config.block_size), dtype=torch.long)
    logits, _ = tiny_model(exact)

    assert logits.shape == (1, tiny_config.block_size, tiny_config.vocab_size)
    with pytest.raises(ValueError, match="exceeds block_size"):
        tiny_model(torch.randint(tiny_config.vocab_size, (1, tiny_config.block_size + 1)))
