from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.byteseed.config import load_config
from src.byteseed.generate import load_model, marker_id, stop_token_ids
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
) -> torch.Tensor:
    with torch.inference_mode():
        return model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            vocab_limit=tokenizer.vocab_size,
            stop_token_ids=stops,
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Benchmark ByteSeed generation latency and throughput.")
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/anchor_v2_3_finetuned.pt")
    parser.add_argument("--prompt", default="what is a stack ?")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16"), default="auto")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile on the model forward pass. Experimental and off by default.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    model = load_model(cfg, args.checkpoint, tokenizer=tokenizer)
    device = next(model.parameters()).device
    dtype_name = resolve_dtype(args.dtype, device)
    model = apply_inference_dtype(model, dtype_name)
    compiled = maybe_compile_forward(model, args.compile)
    stops = stop_token_ids(tokenizer, marker_id(tokenizer, "<|end|>") is not None)
    prompt = build_prompt(args.prompt)
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)

    for _ in range(max(0, args.warmup_runs)):
        synchronize_if_cuda(device)
        _ = run_generation(model, ids, tokenizer, stops, args.max_new_tokens, args.temperature, args.top_k)
        synchronize_if_cuda(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies: list[float] = []
    generated_counts: list[int] = []
    for _ in range(args.runs):
        synchronize_if_cuda(device)
        start = time.perf_counter()
        out = run_generation(model, ids, tokenizer, stops, args.max_new_tokens, args.temperature, args.top_k)
        synchronize_if_cuda(device)
        elapsed = time.perf_counter() - start
        generated = int(out.shape[1] - ids.shape[1])
        latencies.append(elapsed)
        generated_counts.append(generated)

    total_time = sum(latencies)
    total_tokens = sum(generated_counts)
    avg_latency = total_time / max(1, len(latencies))
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    avg_tokens = total_tokens / max(1, len(generated_counts))
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device.type}")
    print(f"dtype used: {dtype_name}")
    print(f"compile: {'on' if compiled else 'off'}")
    print(f"prompt: {args.prompt}")
    print(f"runs: {args.runs}")
    print(f"warmup runs: {args.warmup_runs}")
    print(f"average latency: {avg_latency:.4f} sec")
    print(f"min latency: {min_latency:.4f} sec")
    print(f"max latency: {max_latency:.4f} sec")
    print(f"average generated tokens: {avg_tokens:.1f}")
    print(f"tokens/sec: {tokens_per_sec:.2f}")
    if device.type == "cuda":
        print(f"peak CUDA memory: {peak_memory:.2f} MiB")
    else:
        print("peak CUDA memory: n/a")


if __name__ == "__main__":
    main()
