from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.byteseed.config import load_config
from src.byteseed.generate import load_model, marker_id, stop_token_ids
from src.byteseed.tokenizer import ByteSeedTokenizer


DEFAULT_PROMPTS = [
    "who are you?",
    "Tell me about yourself.",
    "Help me plan a 1 hour DSA study session.",
    "What is a stack?",
    "What is overfitting?",
    "How do I run ByteSeed chat?",
    "My PyTorch says CUDA is false. What should I check?",
    "Should I upload checkpoints to GitHub?",
]


def build_prompt(user_prompt: str) -> str:
    return f"<|user|>\n{user_prompt}\n<|assistant|>\n"


def clean_output(text: str) -> str:
    for marker in ("<|end|>", "<|user|>", "<|assistant|>"):
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return text.strip()


def generate_answer(
    model: torch.nn.Module,
    tokenizer: ByteSeedTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    device = next(model.parameters()).device
    formatted = build_prompt(prompt)
    ids = torch.tensor([tokenizer.encode(formatted, add_bos=True)], dtype=torch.long, device=device)
    stops = stop_token_ids(tokenizer, marker_id(tokenizer, "<|end|>") is not None)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        vocab_limit=tokenizer.vocab_size,
        stop_token_ids=stops,
    )
    text = tokenizer.decode(out[0, ids.shape[1] :].tolist())
    return clean_output(text)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate a ByteSeed chat checkpoint on fixed prompts.")
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/chat_finetuned.pt")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    model = load_model(cfg, args.checkpoint)

    print(f"checkpoint: {args.checkpoint}")
    print(f"temperature: {args.temperature:g} | top_k: {args.top_k} | max_new_tokens: {args.max_new_tokens}")
    print()
    for prompt in DEFAULT_PROMPTS:
        answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k)
        print(f"USER: {prompt}")
        print(f"BYTESEED: {answer if answer else '[empty]'}")
        print()


if __name__ == "__main__":
    main()
