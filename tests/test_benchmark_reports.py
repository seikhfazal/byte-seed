from __future__ import annotations

import json

import pytest
import torch

from byteseed.benchmarking import (
    BenchmarkConfig,
    BenchmarkRun,
    BenchmarkValidationError,
    build_benchmark_report,
    load_benchmark_report,
    measure_generation,
    validate_benchmark_report,
    write_benchmark_report,
)
from byteseed.provenance import canonical_sha256
from scripts.benchmark_generation import synchronize_if_cuda


ENVIRONMENT = {
    "python_version": "test",
    "pytorch_version": "test",
    "device": "cpu",
    "dtype": "fp32",
    "compile": False,
    "deterministic_algorithms_enabled": False,
}


def _config(**overrides):
    values = {
        "seed": 11,
        "warmup_runs": 1,
        "measured_runs": 2,
        "temperature": 0.2,
        "top_k": 5,
        "max_new_tokens": 4,
        "device": "cpu",
        "dtype": "fp32",
        "prompt_digest": "a" * 64,
        "input_token_count": 3,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def _report():
    return build_benchmark_report(
        _config(),
        (
            BenchmarkRun(1, 0.5, 4),
            BenchmarkRun(2, 1.0, 3),
        ),
        checkpoint_identity={"logical_name": "synthetic.pt", "kind": "sft"},
        model_configuration={"n_layer": 1},
        parameter_count=17,
        tokenizer_identity={"logical_name": "synthetic.model"},
        peak_cuda_memory_mib=None,
        environment=ENVIRONMENT,
    )


def test_warmups_are_excluded_and_measured_order_is_deterministic():
    calls: list[str] = []
    clock_values = iter((10.0, 10.5, 20.0, 21.0))

    def run_once():
        calls.append("run")
        return 4

    def synchronize():
        calls.append("sync")

    def after_warmups():
        calls.append("warmups-complete")

    runs = measure_generation(
        run_once,
        _config(),
        clock=lambda: next(clock_values),
        synchronize=synchronize,
        after_warmups=after_warmups,
    )

    assert len(runs) == 2
    assert [run.index for run in runs] == [1, 2]
    assert [run.elapsed_seconds for run in runs] == [0.5, 1.0]
    assert calls.count("run") == 3
    assert calls.count("sync") == 6
    assert calls.index("warmups-complete") > calls.index("run")


def test_tokens_per_second_and_aggregates_are_exact():
    report = _report()
    aggregate = report["measurements"]["aggregate"]

    assert report["version"] == 1
    assert report["kind"] == "generation_benchmark"
    assert report["benchmark_version"] == 1
    assert aggregate["total_elapsed_seconds"] == 1.5
    assert aggregate["total_generated_tokens"] == 7
    assert aggregate["tokens_per_second"] == pytest.approx(7 / 1.5)
    assert report["measurements"]["peak_cuda_memory_mib"] is None
    validate_benchmark_report(report)


@pytest.mark.parametrize("elapsed", [0.0, -0.1])
def test_zero_or_negative_elapsed_time_fails(elapsed):
    values = iter((1.0, 1.0 + elapsed, 2.0, 3.0))
    with pytest.raises(BenchmarkValidationError, match="elapsed time must be positive"):
        measure_generation(
            lambda: 1,
            _config(warmup_runs=0),
            clock=lambda: next(values),
        )


def test_cpu_report_rejects_cuda_memory_and_cuda_report_requires_it():
    runs = (BenchmarkRun(1, 1.0, 1), BenchmarkRun(2, 1.0, 1))
    with pytest.raises(BenchmarkValidationError, match="must not claim CUDA"):
        build_benchmark_report(
            _config(),
            runs,
            peak_cuda_memory_mib=1.0,
            environment=ENVIRONMENT,
        )
    cuda_report = build_benchmark_report(
        _config(device="cuda"),
        runs,
        peak_cuda_memory_mib=12.5,
        environment={**ENVIRONMENT, "device": "cuda"},
    )
    assert cuda_report["measurements"]["peak_cuda_memory_mib"] == 12.5


def test_cuda_synchronization_path_is_monkeypatchable_without_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(str(device)))
    synchronize_if_cuda(torch.device("cuda"))
    synchronize_if_cuda(torch.device("cpu"))
    assert calls == ["cuda"]


def test_report_digest_future_version_wrong_kind_and_tampering_fail():
    report = _report()
    assert report["digest"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "digest"}
    )

    future = dict(report)
    future["version"] = 2
    with pytest.raises(BenchmarkValidationError, match="future benchmark report version"):
        validate_benchmark_report(future)

    wrong_kind = dict(report)
    wrong_kind["kind"] = "evaluation"
    with pytest.raises(BenchmarkValidationError, match="wrong benchmark report kind"):
        validate_benchmark_report(wrong_kind)

    tampered = json.loads(json.dumps(report))
    tampered["measurements"]["runs"][0]["generated_token_count"] = 2
    with pytest.raises(BenchmarkValidationError):
        validate_benchmark_report(tampered)


def test_configuration_is_separate_from_environment_dependent_measurements():
    report = _report()
    assert "warmup_runs" in report["configuration"]
    assert "measured_runs" in report["configuration"]
    assert "runs" not in report["configuration"]
    assert "runs" in report["measurements"]
    assert "python_version" in report["environment"]


def test_utf8_writing_creates_parent_and_refuses_silent_overwrite(tmp_path):
    report = _report()
    destination = tmp_path / "reports" / "benchmark.json"
    write_benchmark_report(destination, report)

    assert load_benchmark_report(destination) == report
    assert destination.read_bytes().decode("utf-8").endswith("\n")
    with pytest.raises(FileExistsError, match="already exists"):
        write_benchmark_report(destination, report)
    write_benchmark_report(destination, report, overwrite=True)


def test_benchmark_script_contains_no_external_model_comparison():
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_generation.py"
    ).read_text(encoding="utf-8")
    assert "external model" not in source.casefold()
    assert "better than" not in source.casefold()
