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

PREFERRED_CHECKPOINTS = (
    "checkpoints/anchor_v2_3_finetuned.pt",
    "checkpoints/anchor_v2_2_finetuned.pt",
)

PROMPTS = [
    "who are you?",
    "what is a stack ?",
    "What is a queue?",
    "What is overfitting?",
    "What is underfitting?",
    "Help me plan a 1 hour DSA study session.",
    "How do I run ByteSeed chat?",
    "My PyTorch says CUDA is false. What should I check?",
    "Should I upload checkpoints to GitHub?",
]


def default_checkpoint() -> str:
    for checkpoint in PREFERRED_CHECKPOINTS:
        if (ROOT / checkpoint).exists():
            return checkpoint
    return PREFERRED_CHECKPOINTS[-1]


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


def contains_any(text: str, options: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(option in lower for option in options)


def check_answer(prompt: str, answer: str) -> tuple[bool, str]:
    lower = answer.lower().strip()
    if not lower:
        return False, "answer is empty"
    if prompt == "who are you?":
        return ("byteseed" in lower, "contains ByteSeed" if "byteseed" in lower else "missing ByteSeed")
    if prompt == "what is a stack ?":
        ok = "lifo" in lower and "bfs" not in lower
        return ok, "contains LIFO and not BFS" if ok else "must contain LIFO and not contain BFS"
    if prompt == "What is a queue?":
        ok = "fifo" in lower and not ("lifo" in lower and "stack" in lower)
        return ok, "contains FIFO without stack confusion" if ok else "must contain FIFO without stack confusion"
    if prompt == "What is overfitting?":
        ok = "training" in lower and ("validation" in lower or "unseen" in lower or "new data" in lower)
        return ok, "mentions training and validation/unseen/new data" if ok else "must mention training and validation or unseen/new data"
    if prompt == "What is underfitting?":
        simple = contains_any(lower, ("too simple", "not trained enough", "not learn", "too limited"))
        poor_train = "training" in lower or "train" in lower
        poor_val = "validation" in lower
        poor = contains_any(lower, ("poor", "bad", "badly", "high loss"))
        ok = simple and poor and poor_train and poor_val
        return ok, "mentions too simple/not trained enough and poor train/validation performance" if ok else "must mention too simple/not trained enough and poor train/validation performance"
    if prompt == "Help me plan a 1 hour DSA study session.":
        return ("minute" in lower, "contains minutes" if "minute" in lower else "missing minutes")
    if prompt == "How do I run ByteSeed chat?":
        ok = "python chat.py" in lower and "chat.py.py" not in lower
        return ok, "contains python chat.py and not chat.py.py" if ok else "must contain python chat.py and not chat.py.py"
    if prompt == "My PyTorch says CUDA is false. What should I check?":
        ok = "cuda" in lower and ("pytorch" in lower or "torch" in lower or "nvidia" in lower)
        return ok, "mentions CUDA and PyTorch/Torch/NVIDIA" if ok else "must mention CUDA and PyTorch or NVIDIA"
    if prompt == "Should I upload checkpoints to GitHub?":
        ok = "checkpoint" in lower and ("do not commit" in lower or "avoid committing" in lower)
        return ok, "says do not commit/avoid committing checkpoints" if ok else "must say do not commit or avoid committing checkpoints"
    return False, "no check defined"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Stable v0.2 regression evaluation for ByteSeed chat checkpoints.")
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to newest stable Anchor checkpoint found.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    checkpoint = args.checkpoint or default_checkpoint()
    model = load_model(cfg, checkpoint, tokenizer=tokenizer)

    passed = 0
    print(f"checkpoint: {checkpoint}")
    for prompt in PROMPTS:
        answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k)
        ok, reason = check_answer(prompt, answer)
        passed += int(ok)
        print(f"PROMPT: {prompt}")
        print(f"ANSWER: {answer if answer else '[empty]'}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
        print(f"REASON: {reason}")
        print()
    print(f"summary: {passed}/{len(PROMPTS)} passed")


if __name__ == "__main__":
    main()

