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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Benchmark ByteSeed generation latency and throughput.")
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/anchor_v2_2_finetuned.pt")
    parser.add_argument("--prompt", default="what is a stack ?")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    model = load_model(cfg, args.checkpoint)
    device = next(model.parameters()).device
    stops = stop_token_ids(tokenizer, marker_id(tokenizer, "<|end|>") is not None)
    prompt = build_prompt(args.prompt)
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)

    latencies: list[float] = []
    generated_counts: list[int] = []
    for _ in range(args.runs):
        synchronize_if_cuda(device)
        start = time.perf_counter()
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            vocab_limit=tokenizer.vocab_size,
            stop_token_ids=stops,
        )
        synchronize_if_cuda(device)
        elapsed = time.perf_counter() - start
        generated = int(out.shape[1] - ids.shape[1])
        latencies.append(elapsed)
        generated_counts.append(generated)

    total_time = sum(latencies)
    total_tokens = sum(generated_counts)
    avg_latency = total_time / max(1, len(latencies))
    avg_tokens = total_tokens / max(1, len(generated_counts))
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device.type}")
    print(f"prompt: {args.prompt}")
    print(f"runs: {args.runs}")
    print(f"average latency: {avg_latency:.4f} sec")
    print(f"average generated tokens: {avg_tokens:.1f}")
    print(f"tokens/sec: {tokens_per_sec:.2f}")


if __name__ == "__main__":
    main()
