from __future__ import annotations

import pytest
import torch


def test_token_embedding_and_lm_head_share_the_tied_parameter(tiny_model):
    embedding = tiny_model.token_embedding.weight
    output = tiny_model.lm_head.weight

    assert embedding is output
    assert embedding.data_ptr() == output.data_ptr()
    with torch.no_grad():
        embedding[0, 0] = 0.1234
    assert output[0, 0].item() == pytest.approx(0.1234)
