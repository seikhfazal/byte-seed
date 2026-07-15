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

PROMPTS = [
    "who are you?",
    "what is a stack ?",
    "What is overfitting?",
    "What is underfitting?",
    "My PyTorch says CUDA is false. What should I check?",
    "How do I run ByteSeed chat?",
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


def has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def check_answer(prompt: str, answer: str) -> tuple[bool, str]:
    lower = answer.lower().strip()
    if not lower:
        return False, "answer is empty"
    if prompt == "who are you?":
        ok = "byteseed" in lower
        return ok, "contains ByteSeed" if ok else "missing ByteSeed"
    if prompt == "what is a stack ?":
        ok = "lifo" in lower and "bfs" not in lower
        return ok, "contains LIFO and not BFS" if ok else "must contain LIFO and not BFS"
    if prompt == "What is overfitting?":
        ok = has_any(lower, ("memorizes", "memorize", "memorizing")) and "training" in lower and has_any(lower, ("validation", "new", "unseen"))
        return ok, "mentions memorizing/training and validation/new/unseen" if ok else "must mention memorizing/training and validation/new/unseen"
    if prompt == "What is underfitting?":
        simple = has_any(lower, ("too simple", "not trained enough"))
        train = "training" in lower or "train" in lower
        heldout = has_any(lower, ("validation", "new", "unseen"))
        ok = simple and train and heldout
        return ok, "mentions too simple/not trained enough plus training and validation/new/unseen" if ok else "must mention too simple/not trained enough plus training and validation/new/unseen"
    if prompt == "My PyTorch says CUDA is false. What should I check?":
        useful = "cuda" in lower and has_any(lower, ("pytorch", "torch", "nvidia"))
        forbidden = has_any(lower, ("github", "checkpoint", "commit", "external storage"))
        ok = useful and not forbidden
        return ok, "mentions CUDA and PyTorch/Torch/NVIDIA without GitHub/checkpoint leakage" if ok else "must mention CUDA and PyTorch/NVIDIA and avoid GitHub/checkpoint/commit/external storage"
    if prompt == "How do I run ByteSeed chat?":
        ok = "python chat.py" in lower and "chat.py.py" not in lower
        return ok, "contains python chat.py and not chat.py.py" if ok else "must contain python chat.py and not chat.py.py"
    if prompt == "Should I upload checkpoints to GitHub?":
        ok = "checkpoint" in lower and ("do not commit" in lower or "avoid committing" in lower)
        return ok, "says do not commit/avoid committing checkpoints" if ok else "must say do not commit or avoid committing checkpoints"
    return False, "no check defined"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate Anchor v2.3 targeted patch behavior.")
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/anchor_v2_3_finetuned.pt")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    model = load_model(cfg, args.checkpoint, tokenizer=tokenizer)

    passed = 0
    print(f"checkpoint: {args.checkpoint}")
    for prompt in PROMPTS:
        answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k)
        ok, reason = check_answer(prompt, answer)
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} | {prompt}")
        print(f"REASON: {reason}")
        print(f"ANSWER: {answer if answer else '[empty]'}")
        print()
    print(f"summary: {passed}/{len(PROMPTS)} passed")


if __name__ == "__main__":
    main()
