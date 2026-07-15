from __future__ import annotations

import pytest
import torch


def test_all_ignored_targets_are_rejected(tiny_config, tiny_model):
    tokens = torch.randint(tiny_config.vocab_size, (1, 4), dtype=torch.long)
    ignored = torch.full((1, 4), -100, dtype=torch.long)

    with pytest.raises(ValueError, match="no supervised target tokens"):
        tiny_model(tokens, ignored)
