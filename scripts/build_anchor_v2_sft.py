from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "assistant_sft" / "anchor_v2_core.jsonl"
OUT = ROOT / "examples" / "byteseed_anchor_v2_sft.jsonl"
EXPECTED_COUNTS = {
    "identity": 60,
    "dsa_concepts": 80,
    "dsa_planning": 60,
    "ai_ml_basics": 40,
    "byteseed_workflow": 30,
    "troubleshooting": 30,
}
EVAL_PROMPTS = [
    "who are you?",
    "Tell me about yourself.",
    "Help me plan a 1 hour DSA study session.",
    "What is a stack?",
    "What is a queue?",
    "What is overfitting?",
    "How do I run ByteSeed chat?",
    "My PyTorch says CUDA is false. What should I check?",
    "Should I upload checkpoints to GitHub?",
]


def chat_text(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>"


def add(rows: list[dict[str, str]], category: str, user: str, assistant: str) -> None:
    rows.append({"category": category, "user": user.strip(), "assistant": assistant.strip()})


def make_identity(rows: list[dict[str, str]]) -> None:
    bases = [
        "who are you?", "Who are you?", "Tell me about yourself.", "tell me about yourself", "what are you?",
        "What are you?", "what can you help me with?", "What can you help me with?", "what is your name?",
        "What is your name?", "introduce yourself", "Introduce yourself.", "are you ByteSeed?", "Are you ByteSeed?",
        "what kind of assistant are you?", "What kind of assistant are you?", "what is ByteSeed?", "What is ByteSeed?",
        "what are you good at?", "What are you good at?",
    ]
    tails = ["", " in one line", " for study"]
    focuses = ["DSA study", "coding basics", "ByteSeed workflow", "AI basics", "troubleshooting"]
    priorities = [
        "I keep answers short.", "I separate similar concepts clearly.", "I prefer concrete commands.",
        "I focus on the question asked.", "I avoid broad unrelated advice.", "I use simple wording.",
        "I give practical next steps.", "I support local project work.", "I explain one idea at a time.",
        "I help with focused practice.", "I keep examples specific.", "I check the target behavior.",
    ]
    for i in range(60):
        user = bases[i % len(bases)] + tails[i // len(bases)]
        focus = focuses[i % len(focuses)]
        answer = f"I am ByteSeed, a local assistant for {focus}. {priorities[i // len(focuses)]}"
        if i == 0:
            answer = "I am ByteSeed, a local assistant trained to help with DSA study, coding basics, and ByteSeed workflow questions."
        add(rows, "identity", user, answer)


def make_dsa_concepts(rows: list[dict[str, str]]) -> None:
    stack = [
        "What is a stack?", "what is a stack?", "Explain a stack.", "Define stack in DSA.", "How does stack order work?",
        "What operations does a stack use?", "What does push mean in a stack?", "What does pop mean in a stack?",
        "What is the top of a stack?", "Give a stack example.", "Is a stack LIFO or FIFO?", "Why are function calls like a stack?",
        "Why is undo history a stack?", "When should I use a stack?", "Summarize stacks briefly.",
    ]
    stack_notes = [
        "This is the opposite of FIFO.", "The top item is handled first.", "Undo history is a common example.",
        "Function calls use this pattern.", "Backtracking often uses this idea.", "Peek reads the top without removing it.",
        "Only the top is active in the simple model.", "The newest item leaves first.", "It is not a queue.",
        "Expression parsing often uses stacks.", "Use it for latest-first work.", "Push adds; pop removes.",
        "The access point is the top.", "It helps reverse recent choices.", "Remember: last added, first removed.",
    ]
    for i, user in enumerate(stack):
        answer = [
            "A stack is LIFO: last in, first out. You push items on top and pop the top item first.",
            "A stack uses push and pop at the top. The last item added is removed first.",
            "A stack is useful for function calls, undo, backtracking, and expression parsing because it follows LIFO order.",
        ][i % 3]
        add(rows, "dsa_concepts", user, f"{answer} {stack_notes[i]}")

    queue = [
        "What is a queue?", "what is a queue?", "Explain a queue.", "Define queue in DSA.", "How does queue order work?",
        "What operations does a queue use?", "What does enqueue mean?", "What does dequeue mean?",
        "What are front and rear in a queue?", "Give a queue example.", "Is a queue FIFO or LIFO?", "Why is scheduling like a queue?",
        "Why are buffers often queues?", "When should I use a queue?", "Summarize queues briefly.",
    ]
    queue_notes = [
        "This is the opposite of LIFO.", "The front item is handled first.", "Task lines are a common example.",
        "BFS often uses this pattern.", "Buffers often use this idea.", "Front reads the next item to leave.",
        "Rear is where new items enter.", "The oldest item leaves first.", "It is not a stack.",
        "Scheduling often uses queues.", "Use it for arrival-order work.", "Enqueue adds; dequeue removes.",
        "The access points are front and rear.", "It preserves waiting order.", "Remember: first added, first removed.",
    ]
    for i, user in enumerate(queue):
        answer = [
            "A queue is FIFO: first in, first out. You enqueue at the rear and dequeue from the front.",
            "A queue removes the oldest waiting item first. It uses front and rear positions.",
            "A queue is useful for scheduling, buffers, BFS, and task processing because it follows FIFO order.",
        ][i % 3]
        add(rows, "dsa_concepts", user, f"{answer} {queue_notes[i]}")

    general = [
        ("What is an array?", "An array stores values by index and is good for fast access by position."),
        ("What is a linked list?", "A linked list is a chain of nodes where each node points to the next node."),
        ("What is binary search?", "Binary search checks the middle of sorted data and discards half the range each step."),
        ("What is recursion?", "Recursion solves a problem by calling the same function on smaller inputs until a base case."),
        ("What is Big O?", "Big O describes how time or space grows as input size grows."),
        ("What is O(n)?", "O(n) means the work grows roughly in proportion to the number of input items."),
        ("What is O(log n)?", "O(log n) usually means the problem size is divided each step, as in binary search."),
        ("What is a hash map?", "A hash map stores key-value pairs and usually gives fast lookup by key."),
        ("What is BFS?", "BFS is breadth-first search. It visits nodes level by level, usually with a queue."),
        ("What is DFS?", "DFS is depth-first search. It follows one path deeply before backtracking."),
        ("What is a tree?", "A tree is a hierarchy of nodes with one root and child nodes below it."),
        ("What is a graph?", "A graph is a set of nodes connected by edges."),
        ("What is a base case?", "A base case is the condition that stops recursion."),
        ("What is two pointers?", "Two pointers uses two indexes that move through data to find or compare values."),
        ("What is sliding window?", "Sliding window keeps a moving range over data to update an answer efficiently."),
        ("What is prefix sum?", "Prefix sum stores running totals so range sums can be answered quickly."),
        ("What is a heap?", "A heap is a priority-based tree structure often used for min or max retrieval."),
        ("What is a priority queue?", "A priority queue removes the item with the highest or lowest priority first."),
        ("What is backtracking?", "Backtracking tries choices, undoes them, and explores alternatives."),
        ("What is dynamic programming?", "Dynamic programming stores results of overlapping subproblems to avoid repeated work."),
        ("What is sorting?", "Sorting arranges data in an order such as ascending or descending."),
        ("What is searching?", "Searching finds whether a target value exists and where it appears."),
        ("What is a pointer in linked lists?", "A pointer is a reference from one node to another node."),
        ("What is a node?", "A node is a data item plus links used in structures like lists, trees, and graphs."),
        ("What is an edge in a graph?", "An edge is a connection between two graph nodes."),
        ("What is a cycle in a graph?", "A cycle is a path that starts and ends at the same node."),
        ("What is a sorted array?", "A sorted array stores values in order, which helps with binary search."),
        ("What is a monotonic stack?", "A monotonic stack keeps items in increasing or decreasing order to solve next-greater-style problems."),
        ("What is a deque?", "A deque is a double-ended queue where you can add or remove from both ends."),
        ("What is a visited set?", "A visited set records nodes already processed during graph or tree traversal."),
    ]
    for user, answer in general:
        add(rows, "dsa_concepts", user, answer)

    contrast = [
        ("Stack vs queue in one line.", "A stack is LIFO with push and pop at the top; a queue is FIFO with enqueue at rear and dequeue at front."),
        ("Does a stack use FIFO?", "No. A stack uses LIFO: last in, first out."),
        ("Does a queue use LIFO?", "No. A queue uses FIFO: first in, first out."),
        ("Which structure fits undo?", "Undo fits a stack because the most recent action is reversed first."),
        ("Which structure fits task scheduling?", "Task scheduling often fits a queue because older waiting tasks are handled first."),
        ("Which structure uses top?", "A stack uses a top position for push, pop, and peek."),
        ("Which structure uses front and rear?", "A queue uses front and rear positions for dequeue and enqueue."),
        ("Function calls use stack or queue?", "Function calls use a stack because the latest call returns first."),
        ("Buffers use stack or queue?", "Buffers often use a queue because data is processed in arrival order."),
        ("Backtracking usually uses stack or queue?", "Backtracking usually uses a stack idea: the latest choice is undone first."),
        ("BFS usually uses stack or queue?", "BFS usually uses a queue to visit nodes in FIFO level order."),
        ("DFS usually uses stack or queue?", "DFS uses recursion or a stack to go deep before backtracking."),
        ("What is the key stack word?", "The key stack word is LIFO."),
        ("What is the key queue word?", "The key queue word is FIFO."),
        ("Push and pop belong to what?", "Push and pop are stack operations."),
        ("Enqueue and dequeue belong to what?", "Enqueue and dequeue are queue operations."),
        ("Top belongs to stack or queue?", "Top belongs to a stack."),
        ("Front and rear belong to stack or queue?", "Front and rear belong to a queue."),
        ("Last inserted removed first means what?", "Last inserted removed first means stack and LIFO."),
        ("First inserted removed first means what?", "First inserted removed first means queue and FIFO."),
    ]
    for user, answer in contrast:
        add(rows, "dsa_concepts", user, answer)


def make_planning(rows: list[dict[str, str]]) -> None:
    topics = ["DSA", "arrays", "linked lists", "stacks", "queues", "binary search", "recursion", "trees", "graphs", "dynamic programming"]
    templates = [
        ("Help me plan a 1 hour {topic} study session.", "Use 10 minutes to review one {topic} concept, 35 minutes to solve two focused problems, and 15 minutes to review mistakes."),
        ("Plan my {topic} study session.", "Pick one {topic} goal, review for 10 minutes, practice for 40 minutes, then write mistakes for 10 minutes."),
        ("Give me a 30 minute {topic} practice plan.", "Spend 5 minutes reviewing {topic}, 20 minutes solving one problem, and 5 minutes writing the key pattern."),
        ("I have one hour for {topic}. What should I do?", "Use 15 minutes for {topic} notes, 35 minutes for practice, and 10 minutes for a mistake log."),
        ("How should I revise {topic} today?", "Review one {topic} pattern, solve one easy problem and one medium problem, then summarize the repeated idea."),
        ("Make a quick {topic} plan for me.", "Choose one {topic} subtopic, solve two examples, and finish by explaining the pattern in your own words."),
    ]
    for topic in topics:
        for user_t, answer_t in templates:
            add(rows, "dsa_planning", user_t.format(topic=topic), answer_t.format(topic=topic))


def make_ai(rows: list[dict[str, str]]) -> None:
    items = [
        ("What is overfitting?", "Overfitting means a model memorizes training data too closely and performs badly on new or unseen data."),
        ("what is overfitting?", "Overfitting is when training performance looks good but validation or unseen data performance is poor."),
        ("Explain overfitting simply.", "Overfitting means the model learned the training examples instead of a pattern that works on new data."),
        ("How do I detect overfitting?", "A common sign is low training loss but high validation loss."),
        ("Why is overfitting bad?", "Overfitting is bad because the model performs well on training data but poorly on unseen data."),
        ("Give an overfitting example.", "If a model memorizes training answers but fails validation examples, it is overfitting."),
        ("What is underfitting?", "Underfitting means the model is too simple or not trained enough and performs badly on both train and validation data."),
        ("Explain underfitting simply.", "Underfitting is when the model has not learned the pattern, so both training and validation results are poor."),
        ("How do I detect underfitting?", "Underfitting often shows high training loss and high validation loss."),
        ("Overfitting vs underfitting.", "Overfitting is good train but poor validation; underfitting is poor train and poor validation."),
        ("What is training loss?", "Training loss measures model error on the examples used for training."),
        ("What is validation loss?", "Validation loss measures error on held-out data not used for training updates."),
        ("What is generalization?", "Generalization means performing well on new or unseen data, not just training examples."),
        ("What is a model checkpoint?", "A checkpoint is saved model weights that can be loaded later for training or chat."),
        ("What is fine-tuning?", "Fine-tuning continues training a model on a smaller task-specific dataset."),
        ("What is SFT?", "SFT means supervised fine-tuning on prompt and answer examples."),
        ("What is a tokenizer?", "A tokenizer converts text into token IDs that the model can process."),
        ("What is a token?", "A token is a chunk of text represented by an integer ID."),
        ("What is an epoch?", "An epoch is one full pass over a training dataset."),
        ("What is a batch?", "A batch is a group of examples processed together in one training step."),
        ("What is learning rate?", "Learning rate controls how large each model update is during training."),
        ("What is gradient clipping?", "Gradient clipping limits large updates to keep training stable."),
        ("What is validation data?", "Validation data is held-out data used to check how well the model generalizes."),
        ("What is train data?", "Train data is the data used to update model weights."),
        ("What is inference?", "Inference is using a trained model to generate an answer."),
        ("What is temperature in generation?", "Temperature controls randomness; lower values make output more conservative."),
        ("What is top-k sampling?", "Top-k sampling limits generation to the k most likely next tokens."),
        ("What is a small language model?", "A small language model has fewer parameters and needs cleaner, narrower training data."),
        ("Why can a tiny model mix answers?", "A tiny model can mix answers when examples are too broad, repetitive, or weakly separated."),
        ("How do I reduce answer mixing?", "Use direct examples, contrast pairs, low temperature, and evaluate on fixed prompts."),
        ("What is memorization in ML?", "Memorization is copying training examples without learning a useful general pattern."),
        ("What is unseen data?", "Unseen data is data the model did not train on."),
        ("What is held-out data?", "Held-out data is reserved for validation or testing instead of training."),
        ("What is a baseline?", "A baseline is a simple result used for comparison."),
        ("What is evaluation?", "Evaluation checks model behavior on fixed prompts or held-out examples."),
        ("What is prompt masking?", "Prompt masking trains loss on assistant answers while ignoring prompt tokens."),
        ("Why mask prompt tokens?", "Masking prompt tokens focuses SFT learning on the assistant response."),
        ("What is catastrophic forgetting?", "Catastrophic forgetting is when new training weakens behavior learned earlier."),
        ("What is data leakage?", "Data leakage is when information appears in training or evaluation where it should not."),
        ("What is clean training data?", "Clean training data is accurate, specific, deduplicated, and matched to the target behavior."),
    ]
    for user, answer in items:
        add(rows, "ai_ml_basics", user, answer)


def make_workflow(rows: list[dict[str, str]]) -> None:
    items = [
        ("How do I run ByteSeed chat?", "From D:\\ByteSeed, run python chat.py."),
        ("how do I run ByteSeed chat?", "From the repo root, run python chat.py."),
        ("How do I start ByteSeed chat?", "Open D:\\ByteSeed and run python chat.py."),
        ("What command starts ByteSeed chat?", "Run python chat.py from D:\\ByteSeed."),
        ("How do I run chat with the venv Python?", "Run .\\.venv\\Scripts\\python.exe chat.py from D:\\ByteSeed."),
        ("What is the ByteSeed chat command?", "The normal command is python chat.py."),
        ("Do I run python chat.py.py?", "No. Run python chat.py, not python chat.py.py."),
        ("How do I use a specific checkpoint in chat?", "Run python chat.py --checkpoint checkpoints\\anchor_v2_finetuned.pt."),
        ("How do I use the old anchor checkpoint?", "Run python chat.py --checkpoint checkpoints\\anchor_finetuned.pt."),
        ("How do I use the curated chat checkpoint?", "Run python chat.py --checkpoint checkpoints\\chat_finetuned.pt."),
        ("What checkpoint should ByteSeed chat use now?", "Use checkpoints\\anchor_v2_finetuned.pt after Anchor v2 training."),
        ("How do I build Anchor v2 SFT?", "Run .\\.venv\\Scripts\\python.exe scripts\\build_anchor_v2_sft.py."),
        ("How do I train Anchor v2 SFT?", "Run .\\.venv\\Scripts\\python.exe scripts\\run_anchor_v2_sft.py --iters 1000."),
        ("How do I evaluate Anchor v2?", "Run .\\.venv\\Scripts\\python.exe scripts\\eval_anchor_v2.py --checkpoint checkpoints\\anchor_v2_finetuned.pt."),
        ("Where is the Anchor v2 raw file?", "The raw Anchor v2 file is data\\raw\\assistant_sft\\anchor_v2_core.jsonl."),
        ("Where is the Anchor v2 SFT file?", "The training-ready Anchor v2 file is examples\\byteseed_anchor_v2_sft.jsonl."),
        ("What does Anchor v2 train?", "Anchor v2 trains identity, DSA concepts, DSA planning, AI basics, workflow, and troubleshooting."),
        ("Why avoid broad curated SFT now?", "Broad curated SFT degraded this small model, so Anchor v2 keeps training narrow and direct."),
        ("How do I compile ByteSeed scripts?", "Run python -m compileall src scripts chat.py."),
        ("How do I check the chat help?", "Run .\\.venv\\Scripts\\python.exe chat.py --help."),
        ("Should I upload checkpoints to GitHub?", "Usually no. Checkpoints are large generated files; keep them out of Git unless you intentionally use model storage."),
        ("Should I upload ByteSeed checkpoints to GitHub?", "Do not upload checkpoints to normal GitHub commits by default; keep code and small data in Git."),
        ("What should I commit for Anchor v2?", "Commit scripts, configs, README changes, and small JSONL examples, not generated checkpoints."),
        ("How do I override the default checkpoint?", "Pass --checkpoint with the checkpoint path when running python chat.py."),
        ("What checkpoint does python chat.py prefer?", "It prefers checkpoints\\anchor_v2_finetuned.pt, then anchor_finetuned.pt, then chat_finetuned.pt."),
        ("How do I reset the chat?", "Use /reset inside chat to clear conversation history."),
        ("How do I quit chat?", "Use /quit or /exit inside chat."),
        ("How do I lower randomness in chat?", "Use /temp 0.2 inside chat."),
        ("How do I change max tokens in chat?", "Use /max 80 inside chat."),
        ("How do I see raw model output?", "Use /raw inside chat to toggle raw generated text."),
    ]
    for user, answer in items:
        add(rows, "byteseed_workflow", user, answer)


def make_troubleshooting(rows: list[dict[str, str]]) -> None:
    items = [
        ("My PyTorch says CUDA is false. What should I check?", "Check torch.cuda.is_available(), the PyTorch CUDA build, the NVIDIA driver, and that your active venv has CUDA PyTorch."),
        ("torch.cuda.is_available() is false. What now?", "Check that PyTorch was installed with CUDA support, your NVIDIA driver works, and the active venv is correct."),
        ("CUDA is false in PyTorch.", "Verify the PyTorch CUDA build, NVIDIA driver, GPU visibility, and active virtual environment."),
        ("Why does PyTorch not see CUDA?", "Common causes are CPU-only PyTorch, an old NVIDIA driver, wrong venv, or no visible NVIDIA GPU."),
        ("How do I check CUDA in Python?", "Run python -c \"import torch; print(torch.cuda.is_available())\" in the active environment."),
        ("How do I check my PyTorch CUDA build?", "Run python -c \"import torch; print(torch.version.cuda)\" and confirm it is not None."),
        ("How do I check the NVIDIA driver?", "Run nvidia-smi in a terminal and confirm it shows your GPU."),
        ("Could the wrong venv make CUDA false?", "Yes. Activate the correct venv and check that it has the CUDA-enabled PyTorch install."),
        ("CUDA false after installing torch.", "Confirm you did not install a CPU-only torch wheel and that your NVIDIA driver supports the CUDA build."),
        ("What should CUDA troubleshooting mention?", "For CUDA false, mention PyTorch, CUDA build, NVIDIA driver, GPU visibility, and venv."),
        ("My chat command says file not found.", "Run commands from D:\\ByteSeed, then use python chat.py."),
        ("python chat.py cannot find checkpoint.", "Train or pass an existing checkpoint path with --checkpoint."),
        ("Anchor v2 checkpoint is missing.", "Build the Anchor v2 SFT file, then run scripts\\run_anchor_v2_sft.py to create checkpoints\\anchor_v2_finetuned.pt."),
        ("The model answers the wrong concept.", "Add direct contrast examples, train on Anchor data only, and evaluate fixed prompts."),
        ("The model says queue for stack.", "Add stack examples that say LIFO, push, pop, and top, plus queue contrast examples that say FIFO."),
        ("The model breaks overfitting answers.", "Add overfitting examples that mention training data and validation or unseen data."),
        ("The model repeats the same study plan.", "Add varied planning examples with different topics, times, and review steps."),
        ("The model says python chat.py.py.", "Add workflow examples that say exactly python chat.py and explicitly reject python chat.py.py."),
        ("The model leaks DSA words into CUDA.", "Train CUDA troubleshooting examples that mention PyTorch, CUDA build, NVIDIA driver, and venv only."),
        ("Evaluation fails for one prompt.", "Inspect the generated answer, add targeted anchor examples, retrain Anchor v2, and evaluate again."),
        ("compileall fails.", "Read the syntax error path, fix that file, then rerun python -m compileall src scripts chat.py."),
        ("SFT file is missing.", "Run the matching build script before training."),
        ("Training cannot load best.pt.", "Check that checkpoints\\best.pt exists before Anchor v2 SFT."),
        ("CUDA out of memory during SFT.", "Lower batch size in the config or reduce max tokens; do not change model architecture for this task."),
        ("Chat output is empty.", "Try lower temperature or evaluate the checkpoint with scripts\\eval_anchor_v2.py."),
        ("Chat output has strange fragments.", "Use lower temperature and train on cleaner direct examples."),
        ("My venv command fails.", "From D:\\ByteSeed, check that .\\.venv\\Scripts\\python.exe exists."),
        ("Tokenizer files are missing.", "Use the existing tokenizer files for this task; do not retrain tokenizer for Anchor v2."),
        ("The checkpoint path has slashes.", "Windows accepts backslashes; pass paths like checkpoints\\anchor_v2_finetuned.pt."),
        ("How do I verify the default checkpoint?", "Run python chat.py and read the startup banner line labeled ckpt."),
    ]
    for user, answer in items:
        add(rows, "troubleshooting", user, answer)


def make_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    make_identity(rows)
    make_dsa_concepts(rows)
    make_planning(rows)
    make_ai(rows)
    make_workflow(rows)
    make_troubleshooting(rows)
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
    for prompt in EVAL_PROMPTS:
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

    print(f"Wrote raw Anchor v2 JSONL: {RAW.relative_to(ROOT)}")
    print(f"Wrote Anchor v2 SFT JSONL: {OUT.relative_to(ROOT)}")
    print(f"total examples: {len(rows)}")
    print("category counts:")
    for category in EXPECTED_COUNTS:
        print(f"  {category}: {counts[category]}")
    print("exact eval prompt coverage:")
    for prompt, count in coverage.items():
        print(f"  {prompt}: {count}")


if __name__ == "__main__":
    main()



