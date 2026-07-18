from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.byteseed.benchmarking import (
    BenchmarkConfig,
    build_benchmark_report,
    measure_generation,
    render_benchmark_report,
    write_benchmark_report,
)
from src.byteseed.checkpoint import CheckpointOperation, load_checkpoint
from src.byteseed.config import load_config
from src.byteseed.eval_prompts import ANCHOR_RETENTION_PROMPTS
from src.byteseed.evaluation import logical_checkpoint_identity
from src.byteseed.generate import load_model, marker_id, stop_token_ids
from src.byteseed.provenance import canonical_sha256, sha256_file
from src.byteseed.tokenizer import ByteSeedTokenizer


def build_prompt(user_prompt: str) -> str:
    return f"<|user|>\n{user_prompt}\n<|assistant|>\n"


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_dtype(requested: str, device: torch.device) -> str:
    if requested == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    if requested == "fp16" and device.type != "cuda":
        print("Warning: --dtype fp16 requires CUDA; falling back to fp32.")
        return "fp32"
    return requested


def apply_inference_dtype(model: torch.nn.Module, dtype_name: str) -> torch.nn.Module:
    if dtype_name == "fp16":
        model = model.half()
    else:
        model = model.float()
    model.eval()
    return model


def maybe_compile_forward(model: torch.nn.Module, enabled: bool) -> bool:
    if not enabled:
        return False
    try:
        model.forward = torch.compile(model.forward)  # type: ignore[method-assign]
    except Exception as exc:  # pragma: no cover - local platform dependent.
        print(f"Warning: torch.compile failed; continuing without compile: {exc}")
        return False
    return True


def run_generation(
    model: torch.nn.Module,
    ids: torch.Tensor,
    tokenizer: ByteSeedTokenizer,
    stops: set[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    with torch.inference_mode():
        return model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            vocab_limit=tokenizer.vocab_size,
            stop_token_ids=stops,
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Benchmark ByteSeed generation latency and throughput.")
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/anchor_v2_3_finetuned.pt")
    parser.add_argument("--prompt", default=ANCHOR_RETENTION_PROMPTS[1].text)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16"), default="auto")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile on the model forward pass. Experimental and off by default.")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides={"device": args.device})
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    loaded = load_checkpoint(
        args.checkpoint,
        CheckpointOperation.MODEL_LOAD,
        runtime_tokenizer_identity=tokenizer.identity,
    )
    model = load_model(cfg, args.checkpoint, tokenizer=tokenizer)
    device = next(model.parameters()).device
    dtype_name = resolve_dtype(args.dtype, device)
    model = apply_inference_dtype(model, dtype_name)
    compiled = maybe_compile_forward(model, args.compile)
    stops = stop_token_ids(tokenizer, marker_id(tokenizer, "<|end|>") is not None)
    prompt = build_prompt(args.prompt)
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    config = BenchmarkConfig(
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        measured_runs=args.runs,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=tuple(sorted(stops)),
        device=str(device),
        dtype=dtype_name,
        compile=compiled,
        deterministic_algorithms=args.deterministic_algorithms,
        prompt_format_version=1,
        prompt_id="benchmark.user-prompt",
        prompt_digest=canonical_sha256(
            {"prompt_format_version": 1, "prompt": args.prompt}
        ),
        input_token_count=ids.shape[1],
    )

    def run_once() -> int:
        output = run_generation(
            model,
            ids,
            tokenizer,
            stops,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.repetition_penalty,
        )
        return int(output.shape[1] - ids.shape[1])

    def reset_peak_memory() -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    runs = measure_generation(
        run_once,
        config,
        synchronize=lambda: synchronize_if_cuda(device),
        after_warmups=reset_peak_memory,
    )
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else None
    )
    provenance = loaded.data.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    data_manifest_digest = provenance.get("data_manifest_digest")
    data_manifest_digest = (
        data_manifest_digest if isinstance(data_manifest_digest, str) else None
    )
    kind = loaded.info.kind.value if loaded.info.kind is not None else loaded.info.kind_label
    checkpoint_identity = logical_checkpoint_identity(
        args.checkpoint,
        version=loaded.info.version,
        kind=kind,
        legacy=loaded.info.legacy,
        progress=loaded.info.progress,
        data_manifest_digest=data_manifest_digest,
        artifact_sha256=sha256_file(args.checkpoint),
    )
    model_fields = (
        "model_name", "vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout"
    )
    warnings = [
        "Timing measurements are environment-dependent and are not reproducible across unlike systems."
    ]
    if args.compile and not compiled:
        warnings.append("torch.compile was requested but could not be enabled.")
    report = build_benchmark_report(
        config,
        runs,
        checkpoint_identity=checkpoint_identity,
        model_configuration={
            field: getattr(model.config, field) for field in model_fields
        },
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        tokenizer_identity=(tokenizer.identity if loaded.tokenizer_verified else None),
        peak_cuda_memory_mib=peak_memory,
        warnings=warnings,
    )
    print(f"checkpoint: {Path(args.checkpoint).name}")
    print(render_benchmark_report(report))
    if args.output_json:
        write_benchmark_report(
            args.output_json,
            report,
            overwrite=args.overwrite,
        )
        print(f"Benchmark report written: {Path(args.output_json).name}")


if __name__ == "__main__":
    main()
