from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "assistant_sft" / "anchor_v2_3_patch.jsonl"
OUT = ROOT / "examples" / "byteseed_anchor_v2_3_sft.jsonl"
EXPECTED_COUNTS = {
    "underfitting_overfitting": 35,
    "cuda_pytorch_troubleshooting": 35,
    "identity_dsa_chat_reinforcement": 10,
}
FORBIDDEN_LABELS = (
    "Reinforcement",
    "Command note",
    "Hygiene note",
    "python chat.py.py",
)
CUDA_FORBIDDEN = ("github", "checkpoint", "checkpoints", "commit", "external storage")
CHECK_NUMBER = re.compile(r"\bCheck\s+\d+\b", re.IGNORECASE)


def chat_text(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>"


def add(rows: list[dict[str, str]], category: str, user: str, assistant: str) -> None:
    rows.append({"category": category, "user": user.strip(), "assistant": assistant.strip()})


def make_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    underfitting = [
        ("What is underfitting?", "Underfitting means the model is too simple or not trained enough, so it performs poorly on training data and validation data."),
        ("what is underfitting?", "Underfitting happens when a model is too simple, with poor performance on both training data and new unseen data."),
        ("Explain underfitting.", "Underfitting means the model is not trained enough or is too simple, so training and validation performance are both poor."),
        ("explain underfitting simply", "Underfitting is learning too little: the model does badly on training data and also on validation or new data."),
        ("Define underfitting.", "Underfitting is when a model is too simple to learn the pattern and performs poorly on training and validation data."),
        ("underfitting meaning", "Underfitting means not enough learning, causing poor training performance and poor validation or unseen data performance."),
        ("How do I detect underfitting?", "Underfitting often shows high training loss and high validation loss because the model is too simple or not trained enough."),
        ("Why is underfitting bad?", "Underfitting is bad because the model performs poorly on training data and still performs poorly on new or validation data."),
        ("Give an underfitting example.", "A model that cannot fit training examples and also fails validation examples is underfitting."),
        ("training loss high validation loss high means what?", "High training loss and high validation loss often mean underfitting: the model is too simple or not trained enough."),
        ("model too simple means what?", "A model that is too simple can underfit, performing poorly on training data and validation or unseen data."),
        ("model not trained enough means what?", "If a model is not trained enough, it can underfit and perform poorly on both training and validation data."),
        ("Does underfitting do well on training data?", "No. Underfitting performs poorly on training data and also poorly on validation or new data."),
        ("Does underfitting do well on validation data?", "No. Underfitting usually performs poorly on validation data because it has not learned the training pattern well."),
        ("How is underfitting different from overfitting?", "Underfitting is too simple and poor on training and validation data; overfitting memorizes training data and fails new data."),
        ("underfitting vs overfitting", "Underfitting performs poorly on training and validation data; overfitting memorizes training data and performs poorly on unseen data."),
        ("What causes underfitting?", "Underfitting can come from a model that is too simple, not trained enough, or missing useful features, causing poor train and validation results."),
        ("How do I fix underfitting?", "Fix underfitting by training longer, using a better model or features, and checking that training and validation performance improve."),
    ]
    for user, assistant in underfitting:
        add(rows, "underfitting_overfitting", user, assistant)

    overfitting = [
        ("What is overfitting?", "Overfitting means a model memorizes training data and performs poorly on validation or new unseen data."),
        ("what is overfitting?", "Overfitting happens when the model memorizes training data instead of generalizing to validation or new data."),
        ("Explain overfitting.", "Overfitting means memorizing training data too closely, causing worse performance on validation or unseen data."),
        ("explain overfitting simply", "Overfitting is memorizing training data, so the model struggles on new or validation data."),
        ("Define overfitting.", "Overfitting is when a model memorizes training data and performs poorly on new unseen data."),
        ("overfitting meaning", "Overfitting means training data is memorized while validation or unseen data performance gets worse."),
        ("How do I detect overfitting?", "Overfitting often shows strong training performance but poor validation or new data performance."),
        ("Why is overfitting bad?", "Overfitting is bad because the model memorizes training data and fails on validation or unseen examples."),
        ("Give an overfitting example.", "A model that remembers training answers but misses validation examples is overfitting."),
        ("training loss low validation loss high means what?", "Low training loss with high validation loss often means overfitting: memorizing training data but failing new data."),
        ("Does overfitting do well on new data?", "No. Overfitting memorizes training data and performs poorly on new unseen data."),
        ("Does overfitting do well on validation data?", "No. Overfitting usually performs poorly on validation data because it memorized training data."),
        ("memorizes training data means what?", "Memorizing training data can mean overfitting if validation or unseen data performance is poor."),
        ("How do I reduce overfitting?", "Reduce overfitting by using validation data, cleaner data, regularization, early stopping, or less memorization of training data."),
        ("overfitting vs underfitting", "Overfitting memorizes training data and fails unseen data; underfitting is too simple and fails training plus validation data."),
        ("What is the main overfitting sign?", "The main sign is memorizing training data while validation or new unseen data performance is poor."),
        ("What does poor validation after good training suggest?", "It suggests overfitting: the model memorizes training data but does not generalize to validation or unseen data."),
    ]
    for user, assistant in overfitting:
        add(rows, "underfitting_overfitting", user, assistant)

    cuda_rows = [
        ("My PyTorch says CUDA is false. What should I check?", "Check torch.cuda.is_available(), the PyTorch CUDA build, the NVIDIA driver, and the active venv."),
        ("torch.cuda.is_available() is false. What should I check?", "Check whether PyTorch has a CUDA build, confirm the NVIDIA driver works, and verify the active venv."),
        ("CUDA is false in PyTorch.", "Verify torch.cuda.is_available(), torch.version.cuda, the NVIDIA driver, and that your venv has CUDA-enabled PyTorch."),
        ("PyTorch CUDA false", "Check the PyTorch CUDA build, NVIDIA driver, GPU visibility, and active virtual environment."),
        ("Why does PyTorch say CUDA is unavailable?", "Common causes are CPU-only PyTorch, missing NVIDIA driver support, wrong venv, or an incompatible CUDA build."),
        ("How do I troubleshoot CUDA false?", "Run torch.cuda.is_available(), check torch.version.cuda, run nvidia-smi, and confirm the active venv."),
        ("How do I check PyTorch CUDA support?", "Run python -c \"import torch; print(torch.cuda.is_available()); print(torch.version.cuda)\" in the active venv."),
        ("How do I check the NVIDIA driver for PyTorch?", "Run nvidia-smi, then check torch.cuda.is_available() from the same venv used for ByteSeed."),
        ("Could my venv make CUDA false?", "Yes. The active venv may have CPU-only PyTorch; install a CUDA-enabled PyTorch build in that venv if needed."),
        ("CUDA works elsewhere but PyTorch says false.", "Check the active venv, PyTorch CUDA build, torch.version.cuda, and whether the Python command uses the expected environment."),
        ("PyTorch installed but CUDA false", "You may have installed CPU-only PyTorch. Check torch.version.cuda and reinstall CUDA-enabled PyTorch if needed."),
        ("What command checks CUDA availability?", "Use python -c \"import torch; print(torch.cuda.is_available())\" from the active venv."),
        ("What command checks PyTorch CUDA version?", "Use python -c \"import torch; print(torch.version.cuda)\" and confirm it is not None."),
        ("What command checks NVIDIA GPU?", "Run nvidia-smi and confirm it sees your NVIDIA GPU and driver."),
        ("CUDA false after activating venv", "Inside the active venv, check torch.cuda.is_available(), torch.version.cuda, and whether PyTorch is CUDA-enabled."),
        ("CUDA false on Windows", "On Windows, check NVIDIA driver with nvidia-smi, then check PyTorch CUDA support inside the active venv."),
        ("ByteSeed says CUDA false", "Check torch.cuda.is_available(), PyTorch CUDA build, NVIDIA driver, and active venv before changing training code."),
        ("My RTX GPU is not detected by PyTorch", "Check nvidia-smi, install a CUDA-enabled PyTorch build, and run torch.cuda.is_available() in the active venv."),
        ("PyTorch only sees CPU", "Confirm the NVIDIA driver works, then check whether the installed PyTorch package includes CUDA support."),
        ("How do I know if torch is CPU-only?", "If torch.version.cuda is None, the installed PyTorch build is likely CPU-only."),
        ("Should I reinstall PyTorch for CUDA?", "If torch.version.cuda is None or CUDA is false with a working NVIDIA driver, reinstall CUDA-enabled PyTorch if needed."),
        ("CUDA unavailable in active environment", "Use the active venv to check torch.cuda.is_available(), torch.version.cuda, and the installed PyTorch build."),
        ("PyTorch CUDA troubleshooting steps", "Check torch.cuda.is_available(), torch.version.cuda, nvidia-smi, NVIDIA driver, and active venv."),
        ("What should I inspect when CUDA is false?", "Inspect PyTorch CUDA build, NVIDIA driver, GPU visibility, and whether the current Python is the right venv."),
        ("CUDA false but nvidia-smi works", "If nvidia-smi works, check that PyTorch is CUDA-enabled and installed in the active venv."),
        ("CUDA false after pip install torch", "Check whether pip installed CPU-only PyTorch; torch.version.cuda should show a CUDA version for GPU use."),
        ("How to confirm ByteSeed uses GPU?", "Run a PyTorch CUDA check in the same venv, then start ByteSeed and read the chat banner device line."),
        ("What are common CUDA false causes?", "Common causes include CPU-only PyTorch, wrong venv, outdated NVIDIA driver, or incompatible CUDA build."),
        ("CUDA false in .venv", "Inside .venv, check torch.cuda.is_available(), torch.version.cuda, and reinstall CUDA-enabled PyTorch if needed."),
        ("Do I need NVIDIA driver for PyTorch CUDA?", "Yes. PyTorch CUDA needs a working NVIDIA driver; verify it with nvidia-smi."),
        ("How do I check active venv for CUDA?", "Run where python, then run torch.cuda.is_available() and torch.version.cuda in that same venv."),
        ("CUDA false during ByteSeed chat", "Check PyTorch CUDA support, NVIDIA driver, active venv, and whether torch.cuda.is_available() returns True."),
        ("CUDA false during training", "Before training, check torch.cuda.is_available(), torch.version.cuda, nvidia-smi, and the active venv."),
        ("GPU exists but torch CUDA false", "Check that the installed PyTorch build includes CUDA and that the NVIDIA driver is current enough."),
        ("What if torch.version.cuda is None?", "Install a CUDA-enabled PyTorch build in the active venv if torch.version.cuda is None and you need GPU support."),
    ]
    for user, assistant in cuda_rows:
        add(rows, "cuda_pytorch_troubleshooting", user, assistant)

    reinforcement = [
        ("who are you?", "I am ByteSeed, a small local assistant for DSA study and ByteSeed workflow. I keep answers short."),
        ("what is a stack ?", "A stack is LIFO: last in, first out. Push and pop remove from the top."),
        ("What is a queue?", "A queue is FIFO: first in, first out, with enqueue and dequeue operations."),
        ("Help me plan a 1 hour DSA study session.", "Use 10 minutes to review one concept, 35 minutes to solve two problems, and 15 minutes to review mistakes."),
        ("How do I run ByteSeed chat?", "cd D:/ByteSeed, then run python chat.py."),
        ("Should I upload checkpoints to GitHub?", "Do not commit checkpoints by default. Keep them local or use external storage."),
        ("Tell me about yourself.", "I am ByteSeed, a concise local assistant for DSA, AI basics, and project workflow."),
        ("Give me a 30 minute DSA plan.", "Review for 5 minutes, solve one problem for 20 minutes, and write mistakes for 5 minutes."),
        ("How should ByteSeed answer?", "ByteSeed should answer the current question directly and keep the response short."),
        ("What is the current stable checkpoint?", "The current stable checkpoint is checkpoints/anchor_v2_2_finetuned.pt."),
    ]
    for user, assistant in reinforcement:
        add(rows, "identity_dsa_chat_reinforcement", user, assistant)

    return rows


def write_raw(rows: list[dict[str, str]]) -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with RAW.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with RAW.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            category = str(row.get("category", "")).strip()
            user = str(row.get("user", "")).strip()
            assistant = str(row.get("assistant", "")).strip()
            if category not in EXPECTED_COUNTS:
                raise ValueError(f"Invalid category in {RAW}:{line_no}: {category!r}")
            if not user or not assistant:
                raise ValueError(f"Missing user or assistant in {RAW}:{line_no}")
            rows.append({"category": category, "user": user, "assistant": assistant})
    return rows


def duplicate_values(rows: list[dict[str, str]], field: str) -> list[tuple[str, int]]:
    return [(value, count) for value, count in Counter(row[field] for row in rows).most_common() if count > 1]


def scan_forbidden(rows: list[dict[str, str]]) -> list[str]:
    hits: list[str] = []
    for index, row in enumerate(rows, start=1):
        text = f"{row['user']}\n{row['assistant']}"
        for phrase in FORBIDDEN_LABELS:
            if phrase.lower() in text.lower():
                hits.append(f"line {index}: forbidden phrase {phrase!r}")
        if CHECK_NUMBER.search(text):
            hits.append(f"line {index}: forbidden Check-number label")
        if row["category"] == "cuda_pytorch_troubleshooting":
            lower = text.lower()
            for phrase in CUDA_FORBIDDEN:
                if phrase in lower:
                    hits.append(f"line {index}: CUDA example contains forbidden phrase {phrase!r}")
    return hits


def validate_rows(rows: list[dict[str, str]]) -> None:
    counts = Counter(row["category"] for row in rows)
    for category, expected in EXPECTED_COUNTS.items():
        actual = counts.get(category, 0)
        if actual != expected:
            raise SystemExit(f"Expected {expected} {category} examples, found {actual}.")
    expected_total = sum(EXPECTED_COUNTS.values())
    if len(rows) != expected_total:
        raise SystemExit(f"Expected {expected_total} examples, found {len(rows)}.")
    user_dupes = duplicate_values(rows, "user")
    assistant_dupes = duplicate_values(rows, "assistant")
    if user_dupes:
        raise SystemExit(f"Duplicate user prompts rejected: {user_dupes[:5]}")
    if assistant_dupes:
        raise SystemExit(f"Duplicate assistant responses rejected: {assistant_dupes[:5]}")
    hits = scan_forbidden(rows)
    if hits:
        for hit in hits:
            print(hit)
        raise SystemExit(f"Forbidden scan failed with {len(hits)} hit(s).")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    write_raw(make_rows())
    rows = read_rows()
    validate_rows(rows)
    counts = Counter(row["category"] for row in rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, str]] = []
    with OUT.open("w", encoding="utf-8") as out:
        for row in rows:
            out_row = {
                "user": row["user"],
                "assistant": row["assistant"],
                "text": chat_text(row["user"], row["assistant"]),
                "source": RAW.relative_to(ROOT).as_posix(),
                "category": row["category"],
            }
            out_rows.append(out_row)
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    validate_rows(out_rows)

    print(f"Wrote raw Anchor v2.3 JSONL: {RAW.relative_to(ROOT)}")
    print(f"Wrote Anchor v2.3 SFT JSONL: {OUT.relative_to(ROOT)}")
    print(f"total examples: {len(rows)}")
    print("category counts:")
    for category in EXPECTED_COUNTS:
        print(f"  {category}: {counts[category]}")
    print("forbidden scan: 0 hits")


if __name__ == "__main__":
    main()
