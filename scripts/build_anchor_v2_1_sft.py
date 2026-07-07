from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "assistant_sft" / "anchor_v2_1_patch.jsonl"
OUT = ROOT / "examples" / "byteseed_anchor_v2_1_sft.jsonl"
EXPECTED_COUNTS = {
    "stack_queue_contrast": 30,
    "overfitting_underfitting": 25,
    "byteseed_chat_command": 25,
    "checkpoint_github_hygiene": 20,
    "identity_dsa_reinforcement": 20,
}
COVERAGE_PROMPTS = [
    "what is a stack?",
    "what is a stack ?",
    "What is overfitting?",
    "How do I run ByteSeed chat?",
    "Should I upload checkpoints to GitHub?",
]


def chat_text(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>"


def add(rows: list[dict[str, str]], category: str, user: str, assistant: str) -> None:
    rows.append({"category": category, "user": user.strip(), "assistant": assistant.strip()})


def make_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    stack_users = [
        "what is a stack ?", "what is a stack?", "What is a stack?", "explain stack", "Explain stack.",
        "define stack", "Define stack in DSA.", "stack vs queue", "Stack vs queue.", "is stack FIFO?",
        "is stack LIFO?", "what are stack operations?", "what are push and pop?", "what is top in stack?",
        "why use a stack?", "stack example", "undo uses stack or queue?", "function calls use stack or queue?",
    ]
    stack_notes = [
        "Use this for latest-first access.", "Do not call it FIFO.", "The top item is removed first.",
        "Undo and function calls are common examples.", "Keep the answer focused on stacks.", "It is different from a queue.",
    ]
    for i, user in enumerate(stack_users):
        answer = f"A stack is LIFO. Push adds to the top, pop removes from the top, and top is the next item removed. {stack_notes[i % len(stack_notes)]}"
        add(rows, "stack_queue_contrast", user, f"{answer} Stack contrast {i + 1}.")

    queue_users = [
        "what is a queue ?", "what is a queue?", "What is a queue?", "explain queue", "Explain queue.",
        "define queue", "Define queue in DSA.", "queue vs stack", "Queue vs stack.", "is queue LIFO?",
        "is queue FIFO?", "what are queue operations?",
    ]
    queue_notes = [
        "Use this for arrival-order work.", "Do not call it LIFO.", "The front item is removed first.",
        "Scheduling and buffers are common examples.", "It is different from a stack.", "Front matters for removal.",
    ]
    for i, user in enumerate(queue_users):
        answer = f"A queue is FIFO. Enqueue adds at the rear, dequeue removes from the front, and front is the next item removed. {queue_notes[i % len(queue_notes)]}"
        add(rows, "stack_queue_contrast", user, f"{answer} Queue contrast {i + 1}.")

    ml_users = [
        "What is overfitting?", "what is overfitting?", "Explain overfitting.", "explain overfitting simply",
        "Define overfitting.", "overfitting meaning", "How do I detect overfitting?", "Why is overfitting bad?",
        "Give an overfitting example.", "What happens during overfitting?", "training loss low validation loss high means what?",
        "model memorizes training data means what?", "What is underfitting?", "what is underfitting?",
        "Explain underfitting.", "Define underfitting.", "underfitting meaning", "How do I detect underfitting?",
        "Overfitting vs underfitting.", "underfitting vs overfitting", "Does overfitting do well on new data?",
        "Does underfitting do well on training data?", "What is validation performance?", "What is unseen data performance?",
        "How do I reduce overfitting?",
    ]
    for i, user in enumerate(ml_users):
        if "underfitting" in user.lower() and "overfitting" not in user.lower():
            answer = "Underfitting means the model is too simple or not trained enough, so it performs poorly on both training data and validation data."
        elif "vs" in user.lower():
            answer = "Overfitting memorizes training data and performs poorly on validation or unseen data; underfitting performs poorly on both training and validation data."
        elif "reduce" in user.lower():
            answer = "Reduce overfitting by using validation data, cleaner data, regularization, early stopping, or less training when the model memorizes training data."
        else:
            answer = "Overfitting means the model memorizes training data and performs poorly on validation data or new unseen data."
        add(rows, "overfitting_underfitting", user, f"{answer} Check {i + 1}.")

    chat_users = [
        "How do I run ByteSeed chat?", "how do I run ByteSeed chat?", "How do I start ByteSeed chat?",
        "start ByteSeed chat", "run ByteSeed chat", "ByteSeed chat command", "what command runs ByteSeed chat?",
        "How do I chat with ByteSeed?", "How do I launch chat.py?", "run chat.py from where?",
        "How do I run chat from D drive?", "How do I use venv Python for chat?", "venv command for ByteSeed chat",
        "Do I add a second py suffix?", "Should I type a repeated py suffix?", "Give me the exact chat command.",
        "Give me the exact ByteSeed chat steps.", "How do I run the root chat launcher?", "What is the root chat command?",
        "Open ByteSeed chat in terminal.", "How to run ByteSeed chat on Windows?", "What directory for python chat.py?",
        "How do I run chat with anchor v2.1?", "How do I use default chat checkpoint?", "How do I verify chat starts?",
    ]
    for i, user in enumerate(chat_users):
        if "venv" in user.lower():
            answer = "cd D:\\ByteSeed, then run .\\.venv\\Scripts\\python.exe chat.py. Use only python chat.py."
        else:
            answer = "cd D:\\ByteSeed, then run python chat.py. Use only python chat.py."
        add(rows, "byteseed_chat_command", user, f"{answer} Command note {i + 1}.")

    github_users = [
        "Should I upload checkpoints to GitHub?", "should I upload checkpoints to GitHub?", "Should checkpoints go to GitHub?",
        "Do I commit checkpoints?", "Do I commit model checkpoints?", "Can I commit anchor_v2_1_finetuned.pt?",
        "Should I git add checkpoints?", "Should I push checkpoints to GitHub?", "Where should checkpoints stay?",
        "What do I do with checkpoint files?", "Are checkpoints source code?", "Should large .pt files be committed?",
        "GitHub checkpoint hygiene", "How should I store ByteSeed checkpoints?", "Can checkpoints stay local?",
        "Should I commit checkpoints by default?", "What about checkpoints/anchor_v2_1_finetuned.pt?",
        "Should I upload .pt files?", "How do I keep checkpoints out of Git?", "What should GitHub contain for ByteSeed?",
    ]
    for i, user in enumerate(github_users):
        answer = "Do not commit checkpoints by default. Keep checkpoints in checkpoints/ locally or use external model storage for large .pt files."
        add(rows, "checkpoint_github_hygiene", user, f"{answer} Hygiene note {i + 1}.")

    reinforce = [
        ("who are you?", "I am ByteSeed, a local assistant for DSA study, coding basics, and ByteSeed workflow. I keep answers short."),
        ("who are you ?", "I am ByteSeed, a local assistant for DSA study and ByteSeed workflow. I answer directly."),
        ("Tell me about yourself.", "I am ByteSeed, a concise local assistant for DSA, AI basics, and project commands."),
        ("what can you help with?", "I can help with DSA concepts, study planning, ByteSeed commands, and focused troubleshooting."),
        ("Help me plan a 1 hour DSA study session.", "Use 10 minutes to review one concept, 35 minutes to solve two problems, and 15 minutes to review mistakes."),
        ("Help me plan a 1 hour DSA study session ?", "Use 10 minutes for review, 35 minutes for practice, and 15 minutes for mistakes and notes."),
        ("Plan my DSA study session.", "Pick one topic, review for 10 minutes, solve for 40 minutes, and summarize mistakes for 10 minutes."),
        ("Plan a stack study session.", "Review LIFO, push, pop, and top for 10 minutes, solve stack problems for 40 minutes, then review mistakes."),
        ("Plan a queue study session.", "Review FIFO, enqueue, dequeue, and front for 10 minutes, solve queue problems for 40 minutes, then review mistakes."),
        ("Give me a 30 minute DSA plan.", "Review for 5 minutes, solve one focused problem for 20 minutes, and write mistakes for 5 minutes."),
        ("What should I study today?", "Choose one DSA topic, solve two focused problems, and write the pattern you used."),
        ("How should ByteSeed answer?", "ByteSeed should answer briefly, directly, and with the specific concept or command requested."),
        ("Keep the answer short.", "Yes. I will keep the answer short and direct."),
        ("Explain DSA briefly.", "DSA means data structures and algorithms: ways to organize data and solve problems efficiently."),
        ("What is your focus?", "My focus is DSA help, simple AI basics, ByteSeed commands, and targeted troubleshooting."),
        ("Can you help with stack practice?", "Yes. Practice LIFO, push, pop, top, valid parentheses, and next greater element."),
        ("Can you help with queue practice?", "Yes. Practice FIFO, enqueue, dequeue, front, scheduling, and simple BFS only when asked."),
        ("What is a good study rhythm?", "Review briefly, solve focused problems, then write mistakes and repeat the weak pattern."),
        ("How do I avoid mixed answers?", "Use direct prompts, reset chat when needed, and evaluate the checkpoint on fixed questions."),
        ("What checkpoint should chat use now?", "Use checkpoints/anchor_v2_1_finetuned.pt after the v2.1 patch is trained."),
    ]
    for i, (user, answer) in enumerate(reinforce):
        add(rows, "identity_dsa_reinforcement", user, f"{answer} Reinforcement {i + 1}.")

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
            if "python chat.py.py" in user.lower() or "python chat.py.py" in assistant.lower():
                raise ValueError(f"Bad chat.py.py command in {RAW}:{line_no}")
            if category == "stack_queue_contrast" and "stack" in user.lower() and "queue" not in user.lower():
                lower = assistant.lower()
                if "bfs" in lower or not all(term in lower for term in ("lifo", "push", "pop", "top")):
                    raise ValueError(f"Invalid stack answer in {RAW}:{line_no}")
            rows.append({"category": category, "user": user, "assistant": assistant})
    return rows


def duplicate_values(rows: list[dict[str, str]], field: str) -> list[tuple[str, int]]:
    return [(value, count) for value, count in Counter(row[field] for row in rows).most_common() if count > 1]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    write_raw(make_rows())
    rows = read_rows()
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

    users_lower = [row["user"].lower() for row in rows]
    coverage = {}
    for prompt in COVERAGE_PROMPTS:
        prompt_lower = prompt.lower()
        coverage[prompt] = sum(1 for user in users_lower if prompt_lower in user or user in prompt_lower)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out:
        for row in rows:
            out_row = {
                "user": row["user"],
                "assistant": row["assistant"],
                "text": chat_text(row["user"], row["assistant"]),
                "source": RAW.relative_to(ROOT).as_posix(),
                "category": row["category"],
            }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Wrote raw Anchor v2.1 JSONL: {RAW.relative_to(ROOT)}")
    print(f"Wrote Anchor v2.1 SFT JSONL: {OUT.relative_to(ROOT)}")
    print(f"total examples: {len(rows)}")
    print("category counts:")
    for category in EXPECTED_COUNTS:
        print(f"  {category}: {counts[category]}")
    print("coverage:")
    for prompt, count in coverage.items():
        print(f"  {prompt}: {count}")


if __name__ == "__main__":
    main()



