from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.byteseed.config import load_config
from src.byteseed.generate import load_model, marker_id, stop_token_ids
from src.byteseed.tokenizer import ByteSeedTokenizer

PROMPTS = [
    "who are you?",
    "what is a stack ?",
    "What is overfitting?",
    "How do I run ByteSeed chat?",
    "Should I upload checkpoints to GitHub?",
    "Help me plan a 1 hour DSA study session.",
]
FORBIDDEN = ("Reinforcement", "Stack contrast", "Command note", "Hygiene note")
CHECK_NUMBER = re.compile(r"\bCheck\s+\d+\b")


def build_prompt(user_prompt: str) -> str:
    return f"<|user|>\n{user_prompt}\n<|assistant|>\n"


def clean_output(text: str) -> str:
    for marker in ("<|end|>", "<|user|>", "<|assistant|>"):
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return text.strip()


def generate_answer(model: torch.nn.Module, tokenizer: ByteSeedTokenizer, prompt: str, max_new_tokens: int, temperature: float, top_k: int | None) -> str:
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
    return clean_output(tokenizer.decode(out[0, ids.shape[1] :].tolist()))


def has_forbidden(answer: str) -> bool:
    return any(phrase in answer for phrase in FORBIDDEN) or CHECK_NUMBER.search(answer) is not None


def passes(prompt: str, answer: str) -> bool:
    lower = answer.lower().strip()
    if not lower or has_forbidden(answer):
        return False
    if prompt == "who are you?":
        return "byteseed" in lower
    if prompt == "what is a stack ?":
        return "lifo" in lower and "bfs" not in lower
    if prompt == "What is overfitting?":
        return "training" in lower and ("validation" in lower or "unseen" in lower or "new data" in lower)
    if prompt == "How do I run ByteSeed chat?":
        return "python chat.py" in lower and "chat.py.py" not in lower
    if prompt == "Should I upload checkpoints to GitHub?":
        return ("do not commit checkpoints" in lower or "avoid committing checkpoints" in lower) and "checkpoint" in lower
    if prompt == "Help me plan a 1 hour DSA study session.":
        return "minutes" in lower
    return False


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate Anchor v2.2 checkpoint on cleanup target prompts.")
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/anchor_v2_2_finetuned.pt")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    model = load_model(cfg, args.checkpoint)

    passed = 0
    print(f"checkpoint: {args.checkpoint}")
    for prompt in PROMPTS:
        answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k)
        ok = passes(prompt, answer)
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} | {prompt}")
        print(f"  {answer if answer else '[empty]'}")
    print(f"summary: {passed}/{len(PROMPTS)} passed")


if __name__ == "__main__":
    main()
