from __future__ import annotations

import pytest
import torch


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="Known v0.4 audit defect: all-ignored SFT targets")
def test_all_ignored_targets_produce_a_finite_loss(tiny_config, tiny_model):
    tokens = torch.randint(tiny_config.vocab_size, (1, 4), dtype=torch.long)
    ignored = torch.full((1, 4), -100, dtype=torch.long)

    _, loss = tiny_model(tokens, ignored)

    assert loss is not None and torch.isfinite(loss)
