from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import byteseed.model as model_module
from byteseed.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointKind,
    build_checkpoint,
    validate_training_config,
    training_config_snapshot,
)
from byteseed import chat as chat_module
from byteseed.chat import build_parser
from byteseed.config import ByteSeedConfig, config_from_checkpoint
from byteseed.generate import load_model
from byteseed.model import CausalSelfAttention, GPT


ATOL = 1e-6
RTOL = 1e-5


def _config(base: ByteSeedConfig, backend: str, *, dropout: float = 0.0) -> ByteSeedConfig:
    return replace(base, attention_backend=backend, dropout=dropout)


def _model_pair(tiny_config: ByteSeedConfig) -> tuple[GPT, GPT]:
    manual = GPT(_config(tiny_config, "manual")).eval()
    sdpa = GPT(_config(tiny_config, "sdpa")).eval()
    sdpa.load_state_dict(manual.state_dict())
    return manual, sdpa


def test_default_and_explicit_manual_select_reference_backend(tiny_config):
    assert tiny_config.attention_backend == "manual"
    model = GPT(tiny_config)
    assert model.attention_backend == "manual"
    assert all(block.attn.attention_backend == "manual" for block in model.blocks)


def test_explicit_sdpa_and_auto_select_sdpa_when_available(tiny_config, monkeypatch):
    monkeypatch.setattr(model_module, "sdpa_is_available", lambda: True)

    assert GPT(_config(tiny_config, "sdpa")).attention_backend == "sdpa"
    assert GPT(_config(tiny_config, "auto")).attention_backend == "sdpa"


def test_auto_falls_back_and_explicit_sdpa_fails_when_unavailable(tiny_config, monkeypatch):
    monkeypatch.setattr(model_module, "sdpa_is_available", lambda: False)

    assert GPT(_config(tiny_config, "auto")).attention_backend == "manual"
    with pytest.raises(RuntimeError, match="scaled_dot_product_attention"):
        GPT(_config(tiny_config, "sdpa"))


def test_invalid_backend_and_legacy_config_behavior(tiny_config):
    with pytest.raises(ValueError, match="attention_backend"):
        replace(tiny_config, attention_backend="unknown")

    legacy = dict(tiny_config.__dict__)
    legacy.pop("attention_backend")
    assert config_from_checkpoint(legacy).attention_backend == "manual"


def test_manual_never_invokes_sdpa(tiny_config, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("manual backend invoked SDPA")

    monkeypatch.setattr(model_module.F, "scaled_dot_product_attention", unexpected)
    attention = CausalSelfAttention(_config(tiny_config, "manual")).eval()
    attention(torch.randn(2, 4, tiny_config.n_embd))


def test_state_dict_and_parameter_compatibility(tiny_config):
    manual, sdpa = _model_pair(tiny_config)

    assert sum(p.numel() for p in manual.parameters()) == sum(
        p.numel() for p in sdpa.parameters()
    )
    assert manual.state_dict().keys() == sdpa.state_dict().keys()
    assert any(key.endswith("attn.mask") for key in sdpa.state_dict())
    manual.load_state_dict(sdpa.state_dict())
    sdpa.load_state_dict(manual.state_dict())


@pytest.mark.parametrize(
    ("saved_backend", "runtime_backend"),
    [("manual", "sdpa"), ("sdpa", "manual")],
)
def test_checkpoint_weights_load_between_backends(
    tmp_path, tiny_config, saved_backend, runtime_backend
):
    saved_model = GPT(_config(tiny_config, saved_backend)).eval()
    checkpoint_path = tmp_path / f"{saved_backend}.pt"
    torch.save(
        build_checkpoint(
            CheckpointKind.MODEL_ONLY,
            model_state=saved_model.state_dict(),
            config=saved_model.config.__dict__,
        ),
        checkpoint_path,
    )
    runtime_config = _config(tiny_config, runtime_backend)
    runtime_config.checkpoint_dir = str(tmp_path)

    loaded = load_model(runtime_config, str(checkpoint_path))

    assert loaded.attention_backend == runtime_backend
    assert loaded.state_dict().keys() == saved_model.state_dict().keys()


@pytest.mark.parametrize("batch,time", [(1, 1), (1, 3), (3, 5), (2, 8)])
def test_attention_forward_equivalence(tiny_config, batch, time):
    manual = CausalSelfAttention(_config(tiny_config, "manual")).eval()
    sdpa = CausalSelfAttention(_config(tiny_config, "sdpa")).eval()
    sdpa.load_state_dict(manual.state_dict())
    activations = torch.randn(batch, time, tiny_config.n_embd)

    assert torch.allclose(manual(activations), sdpa(activations), atol=ATOL, rtol=RTOL)


def test_full_model_logits_loss_and_gradients_are_close(tiny_config):
    manual, sdpa = _model_pair(tiny_config)
    tokens = torch.randint(tiny_config.vocab_size, (2, 6))
    targets = torch.randint(tiny_config.vocab_size, (2, 6))

    manual_logits, manual_loss = manual(tokens, targets)
    sdpa_logits, sdpa_loss = sdpa(tokens, targets)
    assert manual_loss is not None and sdpa_loss is not None
    assert torch.allclose(manual_logits, sdpa_logits, atol=ATOL, rtol=RTOL)
    assert torch.allclose(manual_loss, sdpa_loss, atol=ATOL, rtol=RTOL)

    manual_loss.backward()
    sdpa_loss.backward()
    for name in (
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.qkv.bias",
        "blocks.0.attn.proj.weight",
        "blocks.0.attn.proj.bias",
    ):
        manual_grad = dict(manual.named_parameters())[name].grad
        sdpa_grad = dict(sdpa.named_parameters())[name].grad
        assert manual_grad is not None and sdpa_grad is not None
        assert torch.allclose(manual_grad, sdpa_grad, atol=ATOL, rtol=RTOL)


def test_attention_input_gradients_are_close(tiny_config):
    manual = CausalSelfAttention(_config(tiny_config, "manual")).eval()
    sdpa = CausalSelfAttention(_config(tiny_config, "sdpa")).eval()
    sdpa.load_state_dict(manual.state_dict())
    manual_input = torch.randn(2, 5, tiny_config.n_embd, requires_grad=True)
    sdpa_input = manual_input.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(manual_input)

    (manual(manual_input) * upstream).sum().backward()
    (sdpa(sdpa_input) * upstream).sum().backward()

    assert torch.allclose(manual_input.grad, sdpa_input.grad, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_causal_prefix_is_invariant_to_future_inputs(tiny_config, backend):
    attention = CausalSelfAttention(_config(tiny_config, backend)).eval()
    original = torch.randn(1, 6, tiny_config.n_embd)
    changed = original.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])

    assert torch.allclose(
        attention(original)[:, :4],
        attention(changed)[:, :4],
        atol=ATOL,
        rtol=RTOL,
    )


def test_sdpa_dropout_is_training_only(tiny_config, monkeypatch):
    calls: list[tuple[float, bool]] = []
    original = model_module.F.scaled_dot_product_attention

    def recording_sdpa(*args, **kwargs):
        calls.append((kwargs["dropout_p"], kwargs["is_causal"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module.F, "scaled_dot_product_attention", recording_sdpa)
    attention = CausalSelfAttention(_config(tiny_config, "sdpa", dropout=0.25))
    activations = torch.randn(1, 4, tiny_config.n_embd)

    attention.eval()
    attention(activations)
    attention.train()
    attention(activations)

    assert calls == [(0.0, True), (0.25, True)]


def test_zero_dropout_sdpa_training_and_evaluation_are_deterministic(tiny_config):
    attention = CausalSelfAttention(_config(tiny_config, "sdpa", dropout=0.0))
    activations = torch.randn(2, 4, tiny_config.n_embd)

    attention.train()
    first = attention(activations)
    second = attention(activations)
    attention.eval()
    third = attention(activations)

    assert torch.equal(first, second)
    assert torch.equal(second, third)


def test_exact_resume_records_backend_and_rejects_switch(tiny_config):
    manual = training_config_snapshot(
        tiny_config.__dict__, device_type="cpu", amp_enabled=False
    )
    sdpa_config = _config(tiny_config, "sdpa")
    sdpa = training_config_snapshot(
        sdpa_config.__dict__, device_type="cpu", amp_enabled=False
    )

    validate_training_config(manual, dict(manual))
    with pytest.raises(CheckpointCompatibilityError, match="attention_backend"):
        validate_training_config(manual, sdpa)


def test_legacy_exact_resume_metadata_defaults_to_manual(tiny_config):
    current = training_config_snapshot(
        tiny_config.__dict__, device_type="cpu", amp_enabled=False
    )
    legacy = dict(current)
    legacy.pop("attention_backend")

    validate_training_config(legacy, current)


def test_chat_cli_defaults_to_auto_and_accepts_explicit_overrides():
    parser = build_parser("config.yaml", None, "precise")

    assert parser.parse_args([]).attention_backend == "auto"
    assert parser.parse_args(["--attention-backend", "manual"]).attention_backend == "manual"
    assert parser.parse_args(["--attention-backend", "sdpa"]).attention_backend == "sdpa"
    assert parser.parse_args(["--attention-backend", "auto"]).attention_backend == "auto"
    with pytest.raises(SystemExit):
        parser.parse_args(["--attention-backend", "invalid"])


def test_package_chat_entry_point_defaults_to_auto(monkeypatch):
    received = {}
    monkeypatch.setattr(chat_module, "run_chat", lambda args: received.update(vars(args)))

    chat_module.main([], default_checkpoint="synthetic.pt")

    assert received["attention_backend"] == "auto"


def test_chat_auto_backend_resolves_through_model_selection(tiny_config, monkeypatch):
    parser = build_parser("config.yaml", None, "precise")
    backend = parser.parse_args([]).attention_backend

    monkeypatch.setattr(model_module, "sdpa_is_available", lambda: True)
    assert GPT(_config(tiny_config, backend)).attention_backend == "sdpa"

    monkeypatch.setattr(model_module, "sdpa_is_available", lambda: False)
    assert GPT(_config(tiny_config, backend)).attention_backend == "manual"
    assert GPT(
        _config(tiny_config, parser.parse_args(["--attention-backend", "manual"]).attention_backend)
    ).attention_backend == "manual"
    with pytest.raises(RuntimeError, match="scaled_dot_product_attention"):
        GPT(
            _config(tiny_config, parser.parse_args(["--attention-backend", "sdpa"]).attention_backend)
        )


def test_controlled_generation_matches_between_backends(tiny_config):
    manual, sdpa = _model_pair(tiny_config)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    torch.manual_seed(99)
    manual_tokens = manual.generate(prompt.clone(), max_new_tokens=4, top_k=1)
    torch.manual_seed(99)
    sdpa_tokens = sdpa.generate(prompt.clone(), max_new_tokens=4, top_k=1)

    assert torch.equal(manual_tokens, sdpa_tokens)
