from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "assistant_sft" / "anchor_v2_2_cleanup.jsonl"
OUT = ROOT / "examples" / "byteseed_anchor_v2_2_sft.jsonl"
EXPECTED_COUNTS = {
    "identity": 25,
    "stack_queue_contrast": 25,
    "overfitting_underfitting": 20,
    "byteseed_chat_command": 20,
    "checkpoint_github_hygiene": 15,
    "dsa_planning": 15,
}
FORBIDDEN_PHRASES = (
    "Reinforcement",
    "Stack contrast",
    "Command note",
    "Hygiene note",
    "python chat.py.py",
)
COVERAGE_PROMPTS = [
    "who are you?",
    "what is a stack ?",
    "What is overfitting?",
    "How do I run ByteSeed chat?",
    "Should I upload checkpoints to GitHub?",
    "Help me plan a 1 hour DSA study session.",
]


def chat_text(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>"


def add(rows: list[dict[str, str]], category: str, user: str, assistant: str) -> None:
    rows.append({"category": category, "user": user.strip(), "assistant": assistant.strip()})


def make_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    identity = [
        ("who are you?", "I am ByteSeed, a local assistant for DSA study and ByteSeed workflow. I keep answers short."),
        ("who are you ?", "I am ByteSeed, a small local assistant for DSA, AI basics, and project commands."),
        ("Who are you?", "I am ByteSeed, a concise assistant for studying DSA and using this repo."),
        ("what are you?", "I am ByteSeed, a local chat model tuned for short practical answers."),
        ("what are you ?", "I am ByteSeed, a study and coding assistant for this workspace."),
        ("Tell me about yourself.", "I am ByteSeed. I help with DSA concepts, study plans, and ByteSeed commands."),
        ("tell me about yourself", "I am ByteSeed, a local assistant that answers directly and briefly."),
        ("what can you help with?", "I can help with DSA, simple AI terms, ByteSeed commands, and focused troubleshooting."),
        ("What can you help me with?", "I can explain concepts, make short study plans, and give local project commands."),
        ("what is your name?", "My name is ByteSeed."),
        ("What is your name?", "I am ByteSeed."),
        ("introduce yourself", "I am ByteSeed, a concise local assistant for DSA and project workflow."),
        ("Introduce yourself.", "I am ByteSeed. I answer study and workflow questions in a direct way."),
        ("are you ByteSeed?", "Yes. I am ByteSeed, the local assistant for this project."),
        ("Are you ByteSeed?", "Yes. I am ByteSeed."),
        ("what kind of assistant are you?", "I am a practical assistant for DSA study, coding basics, and ByteSeed usage."),
        ("What kind of assistant are you?", "I am a local learning assistant that works best on single-turn questions."),
        ("what is ByteSeed?", "ByteSeed is this local small language model project and chat assistant."),
        ("What is ByteSeed?", "ByteSeed is a local training and chat project for a small GPT-style model."),
        ("what are you good at?", "I am best at short DSA explanations, study plans, and ByteSeed commands."),
        ("What are you good at?", "I am good at direct DSA help, simple AI basics, and local workflow answers."),
        ("keep answers short", "I will keep answers short and focused."),
        ("answer briefly", "I will answer briefly and directly."),
        ("How should ByteSeed answer?", "ByteSeed should answer the current question directly without carrying extra context."),
        ("Do you need chat history?", "No. I work best in stateless single-turn chat by default."),
    ]
    for user, assistant in identity:
        add(rows, "identity", user, assistant)

    stack_queue = [
        ("what is a stack ?", "A stack is LIFO. Push adds to the top, pop removes from the top, and top is the next item removed."),
        ("what is a stack?", "A stack is LIFO: last in, first out. Push and pop happen at the top."),
        ("What is a stack?", "A stack is a LIFO data structure with push, pop, and top operations."),
        ("explain stack", "A stack uses LIFO order. Push puts an item on top, and pop removes the top item."),
        ("Explain stack.", "A stack keeps the newest item on top. It uses LIFO, push, pop, and top."),
        ("define stack", "A stack is LIFO storage where push adds to top and pop removes from top."),
        ("Define stack in DSA.", "In DSA, a stack is LIFO. Its main operations are push, pop, and top."),
        ("stack vs queue", "A stack is LIFO with push and pop at the top; a queue is FIFO with enqueue and dequeue."),
        ("Stack vs queue.", "Stack means LIFO using top, push, and pop. Queue means FIFO using front, enqueue, and dequeue."),
        ("is stack FIFO?", "No. A stack is LIFO. Push adds to the top and pop removes from the top."),
        ("is stack LIFO?", "Yes. A stack is LIFO, with push, pop, and top controlling access."),
        ("what are stack operations?", "Stack operations are push, pop, and top. They follow LIFO order."),
        ("what are push and pop?", "In a stack, push adds to the top and pop removes from the top, following LIFO."),
        ("what is top in stack?", "Top is the stack position where push adds and pop removes the next LIFO item."),
        ("why use a stack?", "Use a stack for LIFO tasks where the most recent item should be handled first with push and pop."),
        ("what is a queue ?", "A queue is FIFO. Enqueue adds at the rear, dequeue removes from the front."),
        ("what is a queue?", "A queue is FIFO: first in, first out. Enqueue adds and dequeue removes from the front side."),
        ("What is a queue?", "A queue is a FIFO data structure with enqueue, dequeue, and front access."),
        ("explain queue", "A queue uses FIFO order. Enqueue adds new items, and dequeue removes the front item."),
        ("Explain queue.", "A queue keeps the oldest item at the front. It uses FIFO, enqueue, and dequeue."),
        ("define queue", "A queue is FIFO storage where enqueue adds and dequeue removes the front item."),
        ("queue vs stack", "A queue is FIFO with enqueue and dequeue; a stack is LIFO with push and pop."),
        ("is queue FIFO?", "Yes. A queue is FIFO, and dequeue removes the front item first."),
        ("is queue LIFO?", "No. A queue is FIFO. Enqueue adds items and dequeue removes from the front."),
        ("what are queue operations?", "Queue operations are enqueue, dequeue, and front. They follow FIFO order."),
    ]
    for user, assistant in stack_queue:
        add(rows, "stack_queue_contrast", user, assistant)

    overfit = [
        ("What is overfitting?", "Overfitting means a model memorizes training data and performs worse on validation or new unseen data."),
        ("what is overfitting?", "Overfitting is when a model memorizes training data instead of generalizing to validation or new data."),
        ("Explain overfitting.", "Overfitting means memorizing training data and doing poorly on validation data or unseen examples."),
        ("explain overfitting simply", "Overfitting is memorizing training data so the model struggles on new or validation data."),
        ("Define overfitting.", "Overfitting is poor generalization caused by memorizing training data too closely."),
        ("overfitting meaning", "Overfitting means training data is memorized while validation or unseen data performance gets worse."),
        ("How do I detect overfitting?", "Look for very good training performance but worse validation or unseen data performance."),
        ("Why is overfitting bad?", "It is bad because the model memorizes training data and fails on new or validation data."),
        ("Give an overfitting example.", "A model that memorizes training answers but misses validation examples is overfitting."),
        ("training loss low validation loss high means what?", "It often means overfitting: the model memorizes training data but performs worse on validation data."),
        ("model memorizes training data means what?", "It can mean overfitting when validation or unseen data performance is worse."),
        ("Does overfitting do well on new data?", "No. Overfitting memorizes training data and usually performs worse on new unseen data."),
        ("What is underfitting?", "Underfitting means the model is too simple or not trained enough and performs poorly on training and validation data."),
        ("what is underfitting?", "Underfitting is weak learning that performs badly on both training data and validation data."),
        ("Explain underfitting.", "Underfitting means the model has not learned enough, so training and validation results are both poor."),
        ("Define underfitting.", "Underfitting is when the model is too limited to learn the training pattern well."),
        ("underfitting meaning", "Underfitting means poor training performance and poor validation performance."),
        ("How do I detect underfitting?", "High training loss and high validation loss often indicate underfitting."),
        ("Overfitting vs underfitting.", "Overfitting memorizes training data and fails validation; underfitting fails both training and validation."),
        ("underfitting vs overfitting", "Underfitting learns too little; overfitting memorizes training data and performs worse on unseen data."),
    ]
    for user, assistant in overfit:
        add(rows, "overfitting_underfitting", user, assistant)

    chat_commands = [
        ("How do I run ByteSeed chat?", "cd D:/ByteSeed, then run python chat.py."),
        ("how do I run ByteSeed chat?", "Open a terminal, run cd D:/ByteSeed, then run python chat.py."),
        ("How do I start ByteSeed chat?", "Use cd D:/ByteSeed, then python chat.py."),
        ("start ByteSeed chat", "From D:/ByteSeed, run python chat.py."),
        ("run ByteSeed chat", "Run cd D:/ByteSeed first, then run python chat.py."),
        ("ByteSeed chat command", "The command is python chat.py from D:/ByteSeed."),
        ("what command runs ByteSeed chat?", "Use python chat.py after changing to D:/ByteSeed."),
        ("How do I chat with ByteSeed?", "Go to D:/ByteSeed and run python chat.py."),
        ("How do I launch chat.py?", "Run cd D:/ByteSeed, then python chat.py."),
        ("run chat.py from where?", "Run chat.py from D:/ByteSeed with python chat.py."),
        ("How do I run chat from D drive?", "Run cd D:/ByteSeed, then run python chat.py."),
        ("How do I use venv Python for chat?", "From D:/ByteSeed, run .\\.venv\\Scripts\\python.exe chat.py."),
        ("venv command for ByteSeed chat", "Use .\\.venv\\Scripts\\python.exe chat.py from D:/ByteSeed."),
        ("Give me the exact chat command.", "cd D:/ByteSeed, then python chat.py."),
        ("Give me the exact ByteSeed chat steps.", "Step one: cd D:/ByteSeed. Step two: python chat.py."),
        ("How do I run the root chat launcher?", "The root launcher runs from D:/ByteSeed with python chat.py."),
        ("What is the root chat command?", "The root chat command is python chat.py."),
        ("Open ByteSeed chat in terminal.", "In terminal, run cd D:/ByteSeed, then python chat.py."),
        ("How to run ByteSeed chat on Windows?", "On Windows, use cd D:/ByteSeed and then python chat.py."),
        ("What directory for python chat.py?", "Use D:/ByteSeed as the directory, then run python chat.py."),
    ]
    for user, assistant in chat_commands:
        add(rows, "byteseed_chat_command", user, assistant)

    github = [
        ("Should I upload checkpoints to GitHub?", "Do not commit checkpoints by default. Keep them in checkpoints/ locally or use external storage."),
        ("should I upload checkpoints to GitHub?", "Avoid committing checkpoints. Keep checkpoint files local in checkpoints/ or external storage."),
        ("Should checkpoints go to GitHub?", "Do not commit checkpoints by default; they are generated model files."),
        ("Do I commit checkpoints?", "Do not commit checkpoints by default. Keep them local or store them outside Git."),
        ("Do I commit model checkpoints?", "Avoid committing model checkpoints; keep them in checkpoints/ locally or external storage."),
        ("Can I commit anchor_v2_2_finetuned.pt?", "Do not commit checkpoints by default, including anchor_v2_2_finetuned.pt."),
        ("Should I git add checkpoints?", "No. Avoid committing checkpoints unless you intentionally use a model storage workflow."),
        ("Should I push checkpoints to GitHub?", "Do not push checkpoints to GitHub by default; keep them local or external."),
        ("Where should checkpoints stay?", "Keep checkpoints in checkpoints/ locally or use external model storage."),
        ("What do I do with checkpoint files?", "Keep checkpoint files local by default and do not commit them to Git."),
        ("Are checkpoints source code?", "No. Checkpoints are generated model artifacts, not source code."),
        ("Should large .pt files be committed?", "Avoid committing large .pt files; use local checkpoints/ or external storage."),
        ("GitHub checkpoint hygiene", "Checkpoint hygiene means do not commit checkpoints by default."),
        ("How should I store ByteSeed checkpoints?", "Store ByteSeed checkpoints locally in checkpoints/ or in external model storage."),
        ("Can checkpoints stay local?", "Yes. Keep checkpoints local by default and commit only code and small data."),
    ]
    for user, assistant in github:
        add(rows, "checkpoint_github_hygiene", user, assistant)

    plans = [
        ("Help me plan a 1 hour DSA study session.", "Use 10 minutes to review one concept, 35 minutes to solve two problems, and 15 minutes to review mistakes."),
        ("Help me plan a 1 hour DSA study session ?", "Use 10 minutes for review, 35 minutes for practice, and 15 minutes for notes."),
        ("Plan my DSA study session.", "Review for 10 minutes, solve for 40 minutes, and summarize mistakes for 10 minutes."),
        ("Give me a 30 minute DSA plan.", "Review for 5 minutes, solve one problem for 20 minutes, and write mistakes for 5 minutes."),
        ("Plan a stack study session.", "Review LIFO, push, pop, and top for 10 minutes, then solve stack problems for 40 minutes."),
        ("Plan a queue study session.", "Review FIFO, enqueue, dequeue, and front for 10 minutes, then solve queue problems for 40 minutes."),
        ("How should I study stacks?", "Spend 10 minutes on LIFO, push, pop, and top, then practice two stack problems."),
        ("How should I study queues?", "Spend 10 minutes on FIFO, enqueue, dequeue, and front, then practice two queue problems."),
        ("I have one hour for DSA.", "Use 15 minutes for review, 35 minutes for practice, and 10 minutes for mistakes."),
        ("I have one hour for stacks.", "Use 10 minutes for stack basics, 40 minutes for practice, and 10 minutes for review."),
        ("I have one hour for queues.", "Use 10 minutes for queue basics, 40 minutes for practice, and 10 minutes for review."),
        ("Make a quick DSA plan.", "Pick one topic, review briefly, solve one focused problem, then write the pattern."),
        ("What should I study today?", "Choose one DSA topic, solve two focused problems, and review the mistakes."),
        ("How do I review DSA mistakes?", "Write the missed concept, the correct pattern, and one similar problem to retry."),
        ("Give me a simple study rhythm.", "Review, solve, check mistakes, then repeat the weak pattern."),
    ]
    for user, assistant in plans:
        add(rows, "dsa_planning", user, assistant)

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


def forbidden_hits(rows: list[dict[str, str]]) -> list[str]:
    hits: list[str] = []
    check_number = re.compile(r"\bCheck\s+[123]\b")
    for index, row in enumerate(rows, start=1):
        text = f"{row['user']}\n{row['assistant']}"
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text:
                hits.append(f"line {index}: forbidden phrase {phrase!r}")
        if check_number.search(text):
            hits.append(f"line {index}: forbidden numbered Check label")
        if row["category"] == "stack_queue_contrast" and "stack" in row["user"].lower() and "queue" not in row["user"].lower():
            lower = row["assistant"].lower()
            if "bfs" in lower:
                hits.append(f"line {index}: BFS in stack answer")
            if "lifo" not in lower:
                hits.append(f"line {index}: stack answer missing LIFO")
    return hits


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
    hits = forbidden_hits(rows)
    if hits:
        for hit in hits:
            print(hit)
        raise SystemExit(f"Forbidden label scan failed with {len(hits)} hit(s).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    with OUT.open("w", encoding="utf-8") as out:
        for row in rows:
            out_row = {
                "user": row["user"],
                "assistant": row["assistant"],
                "text": chat_text(row["user"], row["assistant"]),
                "source": RAW.relative_to(ROOT).as_posix(),
                "category": row["category"],
            }
            output_rows.append(out_row)
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    output_hits = forbidden_hits(output_rows)
    if output_hits:
        for hit in output_hits:
            print(hit)
        raise SystemExit(f"Forbidden output scan failed with {len(output_hits)} hit(s).")

    users_lower = [row["user"].lower() for row in rows]
    print(f"Wrote raw Anchor v2.2 JSONL: {RAW.relative_to(ROOT)}")
    print(f"Wrote Anchor v2.2 SFT JSONL: {OUT.relative_to(ROOT)}")
    print(f"total examples: {len(rows)}")
    print("category counts:")
    for category in EXPECTED_COUNTS:
        print(f"  {category}: {counts[category]}")
    print("coverage:")
    for prompt in COVERAGE_PROMPTS:
        prompt_lower = prompt.lower()
        count = sum(1 for user in users_lower if prompt_lower in user or user in prompt_lower)
        print(f"  {prompt}: {count}")
    print("forbidden label scan: 0 hits")


if __name__ == "__main__":
    main()

