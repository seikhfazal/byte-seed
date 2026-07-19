from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import byteseed.model as model_module
from byteseed import chat as chat_module
from byteseed.benchmarking import BenchmarkConfig, BenchmarkValidationError
from byteseed.checkpoint import CHECKPOINT_VERSION
from byteseed.generate import (
    build_parser as build_generation_parser,
    validate_generation_execution_options,
)
from byteseed.model import GPT
from scripts.benchmark_generation import (
    build_parser as build_benchmark_parser,
    run_generation as run_benchmark_generation,
)


ATOL = 1e-6
RTOL = 1e-5


def _model(tiny_config, backend: str = "manual") -> GPT:
    return GPT(replace(tiny_config, attention_backend=backend)).eval()


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_prefill_cache_structure_and_logits_match_uncached(tiny_config, backend):
    model = _model(tiny_config, backend)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

    uncached_logits, uncached_loss = model(tokens)
    cached_logits, cache = model.forward_with_cache(tokens)

    assert uncached_loss is None
    assert torch.allclose(cached_logits, uncached_logits, atol=ATOL, rtol=RTOL)
    assert isinstance(cache, tuple)
    assert len(cache) == tiny_config.n_layer
    for key, value in cache:
        assert key.shape == value.shape == (
            tokens.size(0),
            tiny_config.n_head,
            tokens.size(1),
            tiny_config.n_embd // tiny_config.n_head,
        )
        assert key.device == tokens.device
        assert value.device == tokens.device
        assert key.dtype == model.token_embedding.weight.dtype
        assert value.dtype == model.token_embedding.weight.dtype


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_incremental_logits_and_cache_growth_match_full_context(tiny_config, backend):
    model = _model(tiny_config, backend)
    context = torch.tensor([[1, 2, 3], [4, 5, 6]])
    _, cache = model.forward_with_cache(context)

    for token in (torch.tensor([[7], [8]]), torch.tensor([[9], [10]])):
        context = torch.cat((context, token), dim=1)
        cached_logits, cache = model.forward_with_cache(token, cache)
        uncached_logits, _ = model(context)

        assert torch.allclose(
            cached_logits[:, -1],
            uncached_logits[:, -1],
            atol=ATOL,
            rtol=RTOL,
        )
        assert all(key.size(2) == context.size(1) for key, _ in cache)


def test_cached_positions_begin_at_zero_and_advance_from_cache_length(
    tiny_config,
):
    model = _model(tiny_config)
    positions: list[torch.Tensor] = []
    hook = model.position_embedding.register_forward_pre_hook(
        lambda _module, args: positions.append(args[0].detach().clone())
    )
    try:
        _, cache = model.forward_with_cache(torch.tensor([[1, 2, 3]]))
        _, cache = model.forward_with_cache(torch.tensor([[4]]), cache)
        model.forward_with_cache(torch.tensor([[5]]), cache)
    finally:
        hook.remove()

    assert [value.tolist() for value in positions] == [[0, 1, 2], [3], [4]]


def test_cache_validation_rejects_wrong_layer_count_shapes_and_batches(tiny_config):
    model = _model(tiny_config)
    _, cache = model.forward_with_cache(torch.tensor([[1, 2, 3]]))

    with pytest.raises(ValueError, match="layer count"):
        model.forward_with_cache(torch.tensor([[4]]), cache[:-1])

    malformed_shape = ((cache[0][0][..., :-1], cache[0][1]),) + cache[1:]
    with pytest.raises(ValueError, match="shapes must match"):
        model.forward_with_cache(torch.tensor([[4]]), malformed_shape)

    with pytest.raises(ValueError, match="batch size"):
        model.forward_with_cache(torch.tensor([[4], [5]]), cache)

    wrong_heads = (
        (cache[0][0][:, :1], cache[0][1][:, :1]),
    ) + cache[1:]
    with pytest.raises(ValueError, match="attention shape"):
        model.forward_with_cache(torch.tensor([[4]]), wrong_heads)

    shorter = tuple(
        (key[:, :, :-1], value[:, :, :-1]) if index == 1 else (key, value)
        for index, (key, value) in enumerate(cache)
    )
    with pytest.raises(ValueError, match="same sequence length"):
        model.forward_with_cache(torch.tensor([[4]]), shorter)


def test_cache_validation_rejects_dtype_length_and_multi_token_decode(tiny_config):
    model = _model(tiny_config)
    _, cache = model.forward_with_cache(torch.tensor([[1, 2, 3]]))

    wrong_dtype = tuple((key.double(), value.double()) for key, value in cache)
    with pytest.raises(ValueError, match="model dtype"):
        model.forward_with_cache(torch.tensor([[4]]), wrong_dtype)

    head_size = tiny_config.n_embd // tiny_config.n_head
    too_long = tuple(
        (
            torch.zeros(1, tiny_config.n_head, tiny_config.block_size + 1, head_size),
            torch.zeros(1, tiny_config.n_head, tiny_config.block_size + 1, head_size),
        )
        for _ in range(tiny_config.n_layer)
    )
    with pytest.raises(ValueError, match="block_size"):
        model.forward_with_cache(torch.tensor([[4]]), too_long)

    with pytest.raises(ValueError, match="exactly one"):
        model.forward_with_cache(torch.tensor([[4, 5]]), cache)


def test_position_overflow_and_training_mode_fail_clearly(tiny_config):
    model = _model(tiny_config)
    _, cache = model.forward_with_cache(
        torch.arange(tiny_config.block_size).unsqueeze(0)
    )
    with pytest.raises(ValueError, match="invalidate the cache"):
        model.forward_with_cache(torch.tensor([[1]]), cache)

    model.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        model.forward_with_cache(torch.tensor([[1]]))


def test_sdpa_prefill_and_incremental_causal_flags_are_correct(
    tiny_config,
    monkeypatch,
):
    calls: list[bool] = []
    original = model_module.F.scaled_dot_product_attention

    def recording_sdpa(*args, **kwargs):
        calls.append(kwargs["is_causal"])
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module.F, "scaled_dot_product_attention", recording_sdpa)
    model = _model(tiny_config, "sdpa")
    _, cache = model.forward_with_cache(torch.tensor([[1, 2, 3]]))
    model.forward_with_cache(torch.tensor([[4]]), cache)

    assert calls == [True] * tiny_config.n_layer + [False] * tiny_config.n_layer


def test_auto_backend_resolution_is_unchanged_with_cache(
    tiny_config,
    monkeypatch,
):
    monkeypatch.setattr(model_module, "sdpa_is_available", lambda: True)
    sdpa_model = _model(tiny_config, "auto")
    assert sdpa_model.attention_backend == "sdpa"
    sdpa_model.forward_with_cache(torch.tensor([[1, 2]]))

    monkeypatch.setattr(model_module, "sdpa_is_available", lambda: False)
    manual_model = _model(tiny_config, "auto")
    assert manual_model.attention_backend == "manual"
    manual_model.forward_with_cache(torch.tensor([[1, 2]]))


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_cached_greedy_generation_matches_uncached(
    tiny_config,
    backend,
    batch_size,
):
    model = _model(tiny_config, backend)
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])[:batch_size]

    torch.manual_seed(99)
    uncached = model.generate(
        prompt.clone(),
        max_new_tokens=4,
        temperature=0.7,
        top_k=1,
        repetition_penalty=1.2,
    )
    torch.manual_seed(99)
    cached = model.generate(
        prompt.clone(),
        max_new_tokens=4,
        temperature=0.7,
        top_k=1,
        repetition_penalty=1.2,
        use_kv_cache=True,
    )

    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("prompt_length", [1, 7, 8, 10])
def test_rollover_and_context_cropping_match_uncached(tiny_config, prompt_length):
    uncached_model = _model(tiny_config)
    cached_model = _model(tiny_config)
    cached_model.load_state_dict(uncached_model.state_dict())
    prompt = (
        torch.arange(prompt_length, dtype=torch.long).remainder(tiny_config.vocab_size)
        .unsqueeze(0)
    )

    torch.manual_seed(123)
    reference = uncached_model.generate(prompt.clone(), 5, top_k=1)
    torch.manual_seed(123)
    cached = cached_model.generate(
        prompt.clone(),
        5,
        top_k=1,
        use_kv_cache=True,
    )

    assert torch.equal(cached, reference)


def test_rollover_invalidates_cache_and_uses_uncached_path_for_remainder(
    tiny_config,
    monkeypatch,
):
    model = _model(tiny_config)
    prompt = torch.arange(tiny_config.block_size - 1).unsqueeze(0)
    cached_input_lengths: list[int] = []
    uncached_input_lengths: list[int] = []
    original_cached = model.forward_with_cache
    original_forward = model.forward

    def record_cached(idx, past_key_values=None):
        cached_input_lengths.append(idx.size(1))
        return original_cached(idx, past_key_values)

    def record_uncached(idx, targets=None):
        uncached_input_lengths.append(idx.size(1))
        return original_forward(idx, targets)

    monkeypatch.setattr(model, "forward_with_cache", record_cached)
    monkeypatch.setattr(model, "forward", record_uncached)
    model.generate(prompt, 4, top_k=1, use_kv_cache=True)

    assert cached_input_lengths == [tiny_config.block_size - 1, 1]
    assert uncached_input_lengths == [tiny_config.block_size, tiny_config.block_size]


def test_stop_tokens_and_zero_budget_match_uncached(tiny_config):
    model = _model(tiny_config)
    prompt = torch.tensor([[1, 2, 3]])
    first = model.generate(prompt.clone(), 1, top_k=1)
    stop_id = int(first[0, -1])

    uncached = model.generate(
        prompt.clone(), 4, top_k=1, stop_token_ids={stop_id}
    )
    cached = model.generate(
        prompt.clone(),
        4,
        top_k=1,
        stop_token_ids={stop_id},
        use_kv_cache=True,
    )

    assert torch.equal(cached, uncached)
    assert cached.shape[1] == prompt.shape[1] + 1
    assert torch.equal(
        model.generate(prompt.clone(), 0, use_kv_cache=True),
        prompt,
    )


def test_uncached_generation_never_calls_cache_path(tiny_config, monkeypatch):
    model = _model(tiny_config)

    def unexpected(*args, **kwargs):
        raise AssertionError("uncached generation allocated a KV cache")

    monkeypatch.setattr(model, "forward_with_cache", unexpected)
    model.generate(torch.tensor([[1, 2]]), 2, top_k=1)


def test_cache_is_not_retained_between_generation_requests(tiny_config, monkeypatch):
    model = _model(tiny_config)
    calls: list[tuple[int, bool]] = []
    original = model.forward_with_cache

    def recording(idx, past_key_values=None):
        calls.append((idx.size(1), past_key_values is None))
        return original(idx, past_key_values)

    monkeypatch.setattr(model, "forward_with_cache", recording)
    prompt = torch.tensor([[1, 2, 3]])
    model.generate(prompt.clone(), 2, top_k=1, use_kv_cache=True)
    model.generate(prompt.clone(), 2, top_k=1, use_kv_cache=True)

    assert calls == [(3, True), (1, False), (3, True), (1, False)]


def test_cache_does_not_change_weights_state_dict_or_checkpoint_schema(tiny_config):
    model = _model(tiny_config)
    keys_before = tuple(model.state_dict())
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())

    model.forward_with_cache(torch.tensor([[1, 2, 3]]))

    assert tuple(model.state_dict()) == keys_before
    assert not any("cache" in key or "past" in key for key in model.state_dict())
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count_before
    assert CHECKPOINT_VERSION == 1


def test_chat_generation_and_benchmark_cli_cache_defaults_and_overrides():
    chat_parser = chat_module.build_parser("config.yaml", None, "precise")
    generation_parser = build_generation_parser()
    benchmark_parser = build_benchmark_parser()

    assert chat_parser.parse_args([]).kv_cache is False
    assert chat_parser.parse_args(["--kv-cache"]).kv_cache is True
    assert generation_parser.parse_args(["--prompt", "hello"]).kv_cache is False
    assert generation_parser.parse_args(["--prompt", "hello", "--kv-cache"]).kv_cache is True
    assert benchmark_parser.parse_args([]).kv_cache is False
    assert benchmark_parser.parse_args(["--kv-cache"]).kv_cache is True


def test_package_chat_entry_point_preserves_cache_default(monkeypatch):
    received = {}
    monkeypatch.setattr(chat_module, "run_chat", lambda args: received.update(vars(args)))

    chat_module.main([], default_checkpoint="synthetic.pt")
    assert received["kv_cache"] is False

    chat_module.main(["--kv-cache"], default_checkpoint="synthetic.pt")
    assert received["kv_cache"] is True


def test_banner_reports_cache_mode(capsys):
    chat_module.print_banner(
        "test",
        1,
        torch.device("cpu"),
        "synthetic.pt",
        "precise",
        0.2,
        5,
        4,
        False,
        1.0,
        "fp32",
        False,
        "manual",
        True,
    )
    assert "KV cache: on" in capsys.readouterr().out


def test_chat_and_benchmark_helpers_forward_cache_selection():
    class TokenizerDouble:
        vocab_size = 32

        def encode(self, prompt, add_bos):
            assert prompt == "prompt"
            assert add_bos is True
            return [1, 2]

        def decode(self, ids):
            return "response"

    class ModelDouble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.cache_values: list[bool] = []

        def generate(self, ids, **kwargs):
            self.cache_values.append(kwargs["use_kv_cache"])
            return torch.cat((ids, torch.tensor([[3]], device=ids.device)), dim=1)

    model = ModelDouble()
    tokenizer = TokenizerDouble()
    raw, cleaned = chat_module.generate_reply(
        model,
        tokenizer,
        "prompt",
        set(),
        temperature=0.2,
        top_k=1,
        max_new_tokens=1,
        repetition_penalty=1.0,
        use_kv_cache=True,
    )
    assert (raw, cleaned) == ("response", "response")

    run_benchmark_generation(
        model,
        torch.tensor([[1, 2]]),
        tokenizer,
        set(),
        1,
        0.2,
        1,
        use_kv_cache=False,
    )
    assert model.cache_values == [True, False]


def test_compile_and_cache_combination_is_rejected():
    validate_generation_execution_options(
        compile_enabled=False,
        use_kv_cache=True,
    )
    with pytest.raises(ValueError, match="--compile and --kv-cache"):
        validate_generation_execution_options(
            compile_enabled=True,
            use_kv_cache=True,
        )
    with pytest.raises(BenchmarkValidationError, match="compile and kv_cache"):
        BenchmarkConfig(compile=True, kv_cache=True).validate()
