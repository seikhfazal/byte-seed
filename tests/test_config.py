from __future__ import annotations

import pytest

from byteseed.config import ByteSeedConfig, load_config


def test_load_config_coerces_numeric_values(tmp_path):
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        "\n".join(
            [
                'model_name: "Tiny"',
                'vocab_size: "32"',
                'block_size: "8"',
                'n_layer: "2"',
                'n_head: "2"',
                'n_embd: "16"',
                'dropout: "0.0"',
                'batch_size: "2"',
                'gradient_accumulation_steps: "1"',
                'learning_rate: "0.001"',
                'train_split: "0.75"',
                'device: "cpu"',
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.vocab_size == 32
    assert cfg.block_size == 8
    assert cfg.n_embd == 16
    assert cfg.learning_rate == pytest.approx(0.001)
    assert cfg.train_split == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"n_embd": 15, "n_head": 2}, "must be divisible"),
        ({"n_head": 0}, "n_head must be positive"),
        ({"block_size": 0}, "block_size must be positive"),
        ({"vocab_size": 0}, "vocab_size must be positive"),
        ({"train_split": 1.0}, "train_split must be between"),
        ({"train_split": 0.0}, "train_split must be between"),
    ],
)
def test_invalid_configuration_values_raise_clear_errors(values, message):
    with pytest.raises(ValueError, match=message):
        ByteSeedConfig(**values)


def test_invalid_integer_value_has_field_name():
    with pytest.raises(ValueError, match="vocab_size must be an integer-like value"):
        ByteSeedConfig(vocab_size="not-a-number")
