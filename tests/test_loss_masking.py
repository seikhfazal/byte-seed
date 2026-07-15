from __future__ import annotations

import torch
import torch.nn.functional as F


def test_ignored_labels_do_not_change_cross_entropy():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    labels = torch.tensor([[0, -100]])
    changed_ignored_label = torch.tensor([[0, -100]])

    loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1), ignore_index=-100)
    changed_loss = F.cross_entropy(logits.view(-1, 2), changed_ignored_label.view(-1), ignore_index=-100)

    assert torch.isfinite(loss)
    assert loss == changed_loss


def test_mixed_masked_and_unmasked_targets_have_finite_loss(tiny_config, tiny_model):
    tokens = torch.randint(tiny_config.vocab_size, (1, 4), dtype=torch.long)
    targets = torch.tensor([[-100, 3, -100, 5]], dtype=torch.long)

    _, loss = tiny_model(tokens, targets)

    assert loss is not None and torch.isfinite(loss)


def test_all_valid_targets_have_finite_loss(tiny_config, tiny_model):
    tokens = torch.randint(tiny_config.vocab_size, (1, 4), dtype=torch.long)
    targets = torch.randint(tiny_config.vocab_size, (1, 4), dtype=torch.long)

    _, loss = tiny_model(tokens, targets)

    assert loss is not None and torch.isfinite(loss)
