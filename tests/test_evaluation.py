from __future__ import annotations

import random

import pytest
import torch

from byteseed.config import ByteSeedConfig
from byteseed.eval_prompts import (
    ANCHOR_RETENTION_DEFINITION,
    CANDIDATE_PARAPHRASE_DEFINITION,
)
from byteseed.evaluation import (
    GeneratedCaseOutput,
    GenerationConfig,
    run_evaluation,
    torch_batch_generator,
)
from byteseed.model import GPT


ENVIRONMENT = {
    "python_version": "test",
    "pytorch_version": "test",
    "device": "cpu",
    "dtype": "fp32",
    "compile": False,
    "deterministic_algorithms_enabled": False,
}


def _stochastic_generator(cases, _config):
    return [
        GeneratedCaseOutput(
            response=f"{random.random():.12f}:{torch.rand(()).item():.12f}",
            generated_token_count=2,
            stop_reason="max_new_tokens",
        )
        for _ in cases
    ]


def _report(seed: int = 42, *, batch_size: int = 1):
    return run_evaluation(
        CANDIDATE_PARAPHRASE_DEFINITION,
        GenerationConfig(seed=seed, batch_size=batch_size),
        _stochastic_generator,
        checkpoint_identity={"logical_name": "synthetic.pt", "kind": "sft"},
        model_configuration={"n_layer": 1, "n_head": 1},
        parameter_count=7,
        tokenizer_identity={"logical_name": "synthetic.model", "sha256": "0" * 64},
        environment=ENVIRONMENT,
    )


def test_same_seed_produces_identical_stochastic_outputs_and_report_digest():
    first = _report(73)
    second = _report(73)

    assert [item["response"] for item in first["results"]] == [
        item["response"] for item in second["results"]
    ]
    assert first["digest"] == second["digest"]


def test_different_seed_can_change_stochastic_output():
    first = _report(73)
    second = _report(74)
    assert first["results"][0]["response"] != second["results"][0]["response"]


def test_greedy_output_does_not_depend_on_sampling_seed():
    def greedy(cases, _config):
        return [
            GeneratedCaseOutput("fixed", 1, "max_new_tokens") for _ in cases
        ]

    outputs = []
    for seed in (1, 999):
        report = run_evaluation(
            ANCHOR_RETENTION_DEFINITION,
            GenerationConfig(seed=seed, top_k=1, sampling_mode="greedy"),
            greedy,
            environment=ENVIRONMENT,
        )
        outputs.append([item["response"] for item in report["results"]])
    assert outputs[0] == outputs[1]


def test_python_and_torch_cpu_rng_state_are_restored():
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    _report(91)
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_rng_state_is_restored_after_generation_exception():
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()

    def broken(_cases, _config):
        random.random()
        torch.rand(())
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_evaluation(
            CANDIDATE_PARAPHRASE_DEFINITION,
            GenerationConfig(seed=17),
            broken,
            environment=ENVIRONMENT,
        )
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_cpu_evaluation_does_not_query_or_initialize_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden():
        raise AssertionError("CPU evaluation initialized CUDA")

    monkeypatch.setattr(torch.cuda, "is_initialized", forbidden)
    _report(3)


def test_prompt_and_result_order_survive_batched_runner_calls():
    batches: list[list[str]] = []

    def generator(cases, _config):
        batches.append([case.prompt_id for case in cases])
        return [GeneratedCaseOutput(case.prompt_id, 1, "stop_token") for case in cases]

    report = run_evaluation(
        ANCHOR_RETENTION_DEFINITION,
        GenerationConfig(seed=5, batch_size=4),
        generator,
        environment=ENVIRONMENT,
    )

    assert [len(batch) for batch in batches] == [4, 4, 1]
    assert [item["prompt_id"] for item in report["results"]] == [
        case.prompt_id for case in ANCHOR_RETENTION_DEFINITION.cases
    ]
    assert [item["response"] for item in report["results"]] == [
        case.prompt_id for case in ANCHOR_RETENTION_DEFINITION.cases
    ]


def test_generation_configuration_and_stop_metadata_are_explicit():
    report = run_evaluation(
        ANCHOR_RETENTION_DEFINITION,
        GenerationConfig(
            seed=11,
            temperature=0.3,
            top_k=4,
            max_new_tokens=6,
            repetition_penalty=1.1,
            stop_token_ids=(2, 7),
        ),
        lambda cases, _config: [
            GeneratedCaseOutput("answer", 3, "stop_token") for _ in cases
        ],
        environment=ENVIRONMENT,
    )

    assert report["generation"] == {
        "seed": 11,
        "temperature": 0.3,
        "top_k": 4,
        "max_new_tokens": 6,
        "repetition_penalty": 1.1,
        "stop_token_ids": [2, 7],
        "stop_at_end": True,
        "dtype": "fp32",
        "device": "cpu",
        "compile": False,
        "batch_size": 1,
        "prompt_format_version": 1,
        "deterministic_algorithms": False,
        "sampling_mode": "stochastic",
    }
    assert {item["stop_reason"] for item in report["results"]} == {"stop_token"}
    assert {item["generated_token_count"] for item in report["results"]} == {3}


def test_torch_adapter_uses_model_and_tokenizer_doubles_without_artifacts():
    class TokenizerDouble:
        eos_id = 2
        vocab_size = 8

        def encode(self, text, *, add_bos=False, add_eos=False):
            assert text.startswith("<|user|>")
            assert add_bos is True and add_eos is False
            return [1, 3]

        def decode(self, token_ids):
            assert token_ids == [4, 2]
            return "answer<|end|>"

    class ModelDouble:
        def generate(self, idx, **kwargs):
            assert idx.tolist() == [[1, 3]]
            assert kwargs["vocab_limit"] == 8
            assert kwargs["stop_token_ids"] == {2}
            return torch.tensor([[1, 3, 4, 2]])

    generator = torch_batch_generator(ModelDouble(), TokenizerDouble())
    output = generator(
        (ANCHOR_RETENTION_DEFINITION.cases[0],),
        GenerationConfig(stop_token_ids=(2,)),
    )[0]
    assert output.response == "answer"
    assert output.generated_token_count == 2
    assert output.stop_reason == "stop_token"


@pytest.mark.parametrize(
    "suite",
    (ANCHOR_RETENTION_DEFINITION, CANDIDATE_PARAPHRASE_DEFINITION),
)
def test_torch_adapter_uses_real_gpt_generate_and_stops_on_tokenizer_eos(suite):
    class TokenizerDouble:
        eos_id = 2
        vocab_size = 8

        def encode(self, text, *, add_bos=False, add_eos=False):
            assert text.startswith("<|user|>")
            assert add_bos is True and add_eos is False
            return [1, 3]

        def decode(self, token_ids):
            assert token_ids == [self.eos_id]
            return "</s>"

    config = ByteSeedConfig(
        model_name="ByteSeed-Evaluation-Integration",
        vocab_size=8,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=4,
        dropout=0.0,
        device="cpu",
    )
    model = GPT(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.ln_f.weight.fill_(1.0)
        position_signal = torch.tensor([1.0, -1.0, 0.0, 0.0])
        model.position_embedding.weight.copy_(position_signal.repeat(config.block_size, 1))
        model.token_embedding.weight[TokenizerDouble.eos_id].copy_(position_signal)

    outputs = torch_batch_generator(model, TokenizerDouble())(
        suite.cases[:2],
        GenerationConfig(
            seed=17,
            top_k=1,
            max_new_tokens=4,
            stop_token_ids=(),
            stop_at_end=True,
        ),
    )

    assert len(outputs) == 2
    assert all(output.generated_token_count == 1 for output in outputs)
    assert all(output.stop_reason == "stop_token" for output in outputs)
