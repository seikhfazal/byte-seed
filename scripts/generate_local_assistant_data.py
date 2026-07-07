from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw" / "generated"

MARKDOWN_FILES = {
    "byteseed_identity_expanded.md": "ByteSeed Identity Expanded",
    "study_data_structures.md": "Study Data Structures",
    "study_ai_ml_basics.md": "Study AI ML Basics",
    "programming_python.md": "Programming Python",
    "programming_c.md": "Programming C",
    "coding_workflow.md": "Coding Workflow",
    "windows_troubleshooting.md": "Windows Troubleshooting",
    "linux_garuda_troubleshooting.md": "Linux Garuda Troubleshooting",
    "cuda_pytorch_troubleshooting.md": "CUDA PyTorch Troubleshooting",
    "github_repo_hygiene.md": "GitHub Repo Hygiene",
    "byte_seed_project_guide.md": "ByteSeed Project Guide",
    "personal_assistant_routines.md": "Personal Assistant Routines",
    "safe_terminal_commands.md": "Safe Terminal Commands",
    "debugging_conversations.md": "Debugging Conversations",
    "study_planning.md": "Study Planning",
}

CATEGORY_FILES = [
    "generated_personal_assistant.jsonl",
    "generated_study_dsa.jsonl",
    "generated_ai_ml.jsonl",
    "generated_coding_python_c.jsonl",
    "generated_pc_troubleshooting.jsonl",
    "generated_linux_windows.jsonl",
    "generated_byteseed_project.jsonl",
    "generated_github_hygiene.jsonl",
    "generated_safety_boundaries.jsonl",
]

MARKDOWN_CONCEPTS = [
    "planning", "debugging", "tokenizers", "data structures", "Python", "C", "PowerShell", "Linux", "CUDA", "GitHub", "privacy", "checkpoints", "validation", "study habits", "assistant tone",
]
OPENERS = ["Start with", "Use", "Keep", "Inspect", "Split", "Write", "Run", "Review"]
ACTIONS = ["one small step", "a short checklist", "a tiny reproduction", "clear inputs and outputs", "a safe command first", "a focused practice loop", "a plain explanation", "a quick validation"]


def safe_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_markdown(title: str, sections: int, rng: random.Random) -> str:
    parts = [f"# {title}", "", "Synthetic local training notes for ByteSeed. The notes are privacy-safe, educational, and written for a tiny assistant model.", ""]
    for _ in range(sections):
        concept = rng.choice(MARKDOWN_CONCEPTS)
        opener = rng.choice(OPENERS)
        action = rng.choice(ACTIONS)
        parts.extend([
            f"## {concept.title()} practice note",
            "",
            f"- {opener} {action} for {concept}. Keep it local and reversible.",
            "",
            "### Mini sample",
            f"A user is stuck on {concept}; the assistant gives one useful next step and avoids guessing.",
            "",
            "### Command sample",
            "```powershell",
            "python scripts/inspect_dataset.py",
            "```",
            "",
            "## User asks",
            f"I am confused about {concept}. What should I do next?",
            "",
            "## ByteSeed Assistant should respond",
            f"Start small: identify the {concept} problem, run one safe check, and change one thing.",
            "",
            "## Why this response is good",
            "It is direct, practical, honest, and easy for a tiny model to imitate.",
            "",
        ])
    return "\n".join(parts).strip() + "\n"


def personal_examples() -> list[tuple[str, str]]:
    return [
        ("Help me plan a focused 1 hour study session.", "Use 10 minutes to review yesterday's notes, 35 minutes for one hard topic, and 15 minutes to write mistakes and next steps. Keep the phone away and stop when the hour ends."),
        ("I feel lost and do not know what to do next.", "Write the exact goal in one line, then list three small actions. Do the first action only. If it still feels unclear, send me the goal and what is blocking you."),
        ("My tiny model output is repetitive. Should I panic?", "Do not panic. A tiny model often repeats when data is small, templated, or overtrained. Improve the dataset, add stop tokens, lower generation length, and test with short prompts."),
        ("Make me a simple daily coding routine.", "Use a compact routine: 20 minutes reading code, 40 minutes implementing one small change, 20 minutes debugging, and 10 minutes writing what you learned."),
        ("I keep switching tasks and finishing nothing.", "Pick one task for the next 25 minutes. Write the done condition before starting. When the timer ends, either ship the small result or write the exact blocker."),
    ]


def dsa_examples() -> list[tuple[str, str]]:
    return [
        ("Help me plan a 1 hour DSA study session.", "Use 10 minutes for revision, 25 minutes for linked list practice, 15 minutes for stack and queue problems, and 10 minutes to write mistakes. End by naming one pattern to revisit tomorrow."),
        ("Explain arrays versus linked lists.", "Arrays give fast index access and store items contiguously. Linked lists make insertion easier when you already have the node, but finding a position is slower. Use arrays first unless pointer movement is the lesson."),
        ("How should I practice recursion?", "Trace two small examples by hand, write the base case first, then write the recursive step. If the call stack feels confusing, draw each function call as a box."),
        ("Give me a Big O checklist.", "Inspect the input size, count nested loops, look for repeated work, and ask whether a data structure changes lookup cost. Then describe time and space separately."),
        ("I cannot understand graph traversal.", "Start with BFS for shortest unweighted paths and DFS for exploring depth. Draw a visited set, a queue or stack, and update them after every step."),
    ]


def ai_ml_examples() -> list[tuple[str, str]]:
    return [
        ("What is a token in ByteSeed?", "A token is a small text unit the tokenizer maps to an integer. ByteSeed trains on those integers, not raw characters. Special chat markers should stay as whole tokens."),
        ("Explain validation loss simply.", "Validation loss measures how well the model predicts held-out data. If train loss falls but validation loss gets worse, the model is probably memorizing instead of generalizing."),
        ("Why does temperature change output?", "Temperature changes how sharp the probability distribution is. Lower values make output safer and more repetitive. Higher values add variety but can make mistakes more likely."),
        ("What is SFT for this project?", "SFT teaches user-to-assistant behavior after pretraining. For ByteSeed, it should show short, direct answers with stop markers, not long copied templates."),
        ("Why can a tiny model still sound bad after training?", "It has limited capacity and sees limited data. It may memorize common phrases, repeat patterns, or fail at reasoning. Better data and careful generation settings help, but it remains a learning model."),
    ]


def coding_examples() -> list[tuple[str, str]]:
    return [
        ("My Python script cannot find a file path.", "Change the function that opens the file. Print the resolved path with `Path(path).resolve()` before reading it, then verify the file exists from the project root."),
        ("How do I read JSONL in Python?", "Change the loader function. Read one line at a time, skip blank lines, call `json.loads(line)`, and validate required keys before using the row."),
        ("My C linked list crashes when inserting.", "Inspect the insert function first. Verify every allocated node has `next` initialized, handle the empty-list case, and make sure you do not dereference a null pointer."),
        ("Show me a small Python function pattern.", "Put the logic in one function, pass inputs as arguments, and return the result. Keep printing in `main()` so the function is easy to test."),
        ("My C compile error mentions an unknown type.", "Look at the first compiler error, not the last. Inspect whether the struct is declared before use, the header is included, and the typedef name matches exactly."),
    ]


def windows_examples() -> list[tuple[str, str]]:
    return [
        ("PowerShell says my venv activation script is not recognized.", "From the project root, run `.\\.venv\\Scripts\\activate`. If the path still fails, check that `.venv` exists with `Get-ChildItem -Force`. Do not reinstall packages until the path is confirmed."),
        ("How do I check CUDA from Windows?", "Run a harmless Python check: `python -c \"import torch; print(torch.cuda.is_available())\"`. If it prints false, check the PyTorch build, NVIDIA driver, and whether you are inside the right venv."),
        ("Python says torch is missing.", "First activate the venv, then run `python -m pip show torch`. If it is missing, install the correct PyTorch build for your CUDA setup from the official PyTorch selector."),
        ("A checkpoint will not load after tokenizer changes.", "That is expected when vocab size changes. Use a checkpoint trained with the same tokenizer, or retrain tokenizer, prepare data, and train a fresh matching checkpoint."),
        ("PowerShell says access is denied.", "Inspect whether the file is open in another program and whether the command writes outside the project. Avoid force deleting anything until you know the exact path."),
    ]


def linux_examples() -> list[tuple[str, str]]:
    return [
        ("How should I safely update Garuda Linux packages?", "Use the distro's recommended update tool or `sudo pacman -Syu` when you understand the prompt. Read package conflict messages before answering yes."),
        ("How do I inspect a user service log?", "Use `systemctl --user status service-name` first, then `journalctl --user -u service-name --no-pager`. Read the newest error before changing config."),
        ("What is a safe habit for terminal commands?", "Read the command, identify the target path, and avoid recursive delete or force flags until you know what they affect. When unsure, run a listing command first."),
        ("Flatpak app is not launching. What next?", "Run it from the terminal with `flatpak run app.id` to capture the error. Then inspect permissions and recent updates before reinstalling."),
        ("Pacman reports a file conflict.", "Stop and read the conflicting path. Do not overwrite blindly. Inspect which package owns it with `pacman -Qo path` and decide based on that evidence."),
    ]


def byteseed_examples() -> list[tuple[str, str]]:
    return [
        ("What is the clean ByteSeed training workflow?", "Generate or edit local data, inspect it, build the combined corpus, train the tokenizer, prepare token arrays, run a short pretrain test, then run SFT. Change one stage at a time."),
        ("Why retrain the tokenizer after adding special tokens?", "The tokenizer has to learn `<|user|>`, `<|assistant|>`, and `<|end|>` as stable pieces. If you skip retraining, chat formatting may split badly and stopping becomes unreliable."),
        ("Why should checkpoints stay out of GitHub?", "Checkpoint files are large generated artifacts and can change often. Keep source, configs, docs, and small examples in Git; keep checkpoints local or in release storage later."),
        ("Why does ByteSeed repeat templates?", "The synthetic data may contain repeated phrases, and a tiny model can memorize them. Reduce counters, vary answers, add stop markers, and avoid overtraining on one pattern."),
        ("What should I do before a short pretrain run?", "Run dataset inspection, rebuild the corpus, retrain the tokenizer if data changed, prepare data, and count parameters. Then use a limited `--max-iters` test."),
    ]


def github_examples() -> list[tuple[str, str]]:
    return [
        ("What should I keep out of a public GitHub repo?", "Keep out checkpoints, processed arrays, private notes, real secrets, local run logs, and large generated artifacts. Commit source code, configs, docs, and small safe examples."),
        ("How should I write a useful commit message?", "Use a short verb phrase that says what changed, like `Improve generated SFT variety`. If needed, add one body line explaining why the change matters."),
        ("What belongs in the README before publishing?", "Include what the project is, setup steps, basic commands, limits of the model, safety notes, and what generated files should not be committed."),
        ("Why does .gitignore matter for ByteSeed?", "It prevents accidental commits of checkpoints, processed data, runs, local environments, and tokenizer binaries. Review it before publishing."),
        ("Should LICENSE be included?", "Yes. A LICENSE tells others how they may use the code. Keep dataset license notes separate when imported data is added later."),
    ]


def safety_examples() -> list[tuple[str, str]]:
    return [
        ("What should you do if you have not seen my files?", "Say that I have not inspected them yet and ask for the relevant path, snippet, or log. I should not pretend to know local file contents."),
        ("How should you handle destructive commands?", "Warn first, explain the target path and effect, and prefer a listing or dry-run style check before deletion or reset commands."),
        ("Can you pretend you searched the internet?", "No. If I did not browse, I should say so. For current facts, I should use a real lookup or explain that I am working from local context."),
        ("What private data should stay out of examples?", "Do not include real passwords, tokens, API keys, private emails, phone numbers, addresses, or private notes. Use placeholders only when teaching format."),
        ("What if troubleshooting details are missing?", "Ask for the exact command, full error text, environment, and recent change. Guessing is less useful than getting the missing evidence."),
    ]


EXAMPLE_BANKS = {
    "generated_personal_assistant.jsonl": personal_examples,
    "generated_study_dsa.jsonl": dsa_examples,
    "generated_ai_ml.jsonl": ai_ml_examples,
    "generated_coding_python_c.jsonl": coding_examples,
    "generated_pc_troubleshooting.jsonl": windows_examples,
    "generated_linux_windows.jsonl": linux_examples,
    "generated_byteseed_project.jsonl": byteseed_examples,
    "generated_github_hygiene.jsonl": github_examples,
    "generated_safety_boundaries.jsonl": safety_examples,
}

FOLLOWUPS = [
    "Keep it short and practical.",
    "Give me the safest next step.",
    "Explain it like I am debugging alone.",
    "Make it useful for a beginner.",
    "What should I inspect first?",
    "Turn it into a tiny checklist.",
    "Answer in a calm direct style.",
    "Give me a simple plan.",
]
CONTEXTS = [
    "I have about 20 minutes.",
    "I want the lowest-risk option.",
    "I am working from the project root.",
    "I need a beginner-friendly version.",
    "I want to avoid wasted steps.",
    "I am preparing for a short practice run.",
    "I only want local changes.",
    "I need something I can verify quickly.",
]
ANSWER_BRIDGES = [
    "A good way to handle it is:",
    "Use this small plan:",
    "Start with this:",
    "Keep the response focused:",
    "The practical path is:",
    "For a clean first pass:",
    "Use this as the next move:",
    "Treat it as a small debugging task:",
]

ANSWER_SUFFIXES = [
    "If the result is different from expected, capture the exact output before changing anything else.",
    "Keep the change small so you can tell whether it helped.",
    "Write down the mistake you found; that becomes review material.",
    "Stop after the first clear blocker and inspect that evidence.",
    "Use one clean test before moving to the next step.",
    "If you are unsure, ask for the missing log or file path instead of guessing.",
    "Prefer a reversible action before a risky one.",
    "End with the next concrete action, not a vague reminder.",
]


def expanded_examples(filename: str, target: int, rng: random.Random) -> list[dict[str, str]]:
    seeds = EXAMPLE_BANKS[filename]()
    rows: list[dict[str, str]] = []
    seen_users: set[str] = set()
    seen_assistants: set[str] = set()
    prefix_counts: Counter[str] = Counter()
    attempts = 0
    while len(rows) < target and attempts < target * 100:
        attempts += 1
        user, assistant = rng.choice(seeds)
        followup = rng.choice(FOLLOWUPS)
        context = rng.choice(CONTEXTS)
        bridge = rng.choice(ANSWER_BRIDGES)
        suffix = rng.choice(ANSWER_SUFFIXES)
        user_variant = f"{user} {followup} {context}"
        assistant_variant = f"{bridge} {assistant} {suffix}"
        prefix = assistant_variant[:80].lower()
        if user_variant in seen_users or assistant_variant in seen_assistants:
            continue
        if prefix_counts[prefix] >= 80:
            continue
        seen_users.add(user_variant)
        seen_assistants.add(assistant_variant)
        prefix_counts[prefix] += 1
        rows.append({"user": user_variant, "assistant": assistant_variant})
    if len(rows) < target:
        raise RuntimeError(f"Could only create {len(rows)} unique examples for {filename}; add more templates.")
    return rows


def generate(seed: int, examples_per_category: int, markdown_sections_per_topic: int, out_dir: Path, overwrite: bool) -> tuple[int, int, int]:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.rglob("*")) and not overwrite:
        raise SystemExit(f"Output directory already has files: {out_dir}. Use --overwrite to replace generated data.")

    rng = random.Random(seed)
    markdown_dir = out_dir / "markdown"
    sft_dir = out_dir / "sft"

    total_md_chars = 0
    for filename, title in MARKDOWN_FILES.items():
        text = make_markdown(title, markdown_sections_per_topic, rng)
        total_md_chars += len(text)
        safe_write(markdown_dir / filename, text)

    total_examples = 0
    global_users: set[str] = set()
    global_assistants: set[str] = set()
    for filename in CATEGORY_FILES:
        rows = []
        for row in expanded_examples(filename, examples_per_category, rng):
            if row["user"] in global_users or row["assistant"] in global_assistants:
                continue
            global_users.add(row["user"])
            global_assistants.add(row["assistant"])
            rows.append(row)
        if len(rows) != examples_per_category:
            raise RuntimeError(f"Expected {examples_per_category} examples for {filename}, got {len(rows)}")
        total_examples += len(rows)
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        safe_write(sft_dir / filename, text)

    return len(MARKDOWN_FILES), total_md_chars, total_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local synthetic ByteSeed assistant data without internet access.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--examples-per-category", type=int, default=300)
    parser.add_argument("--markdown-sections-per-topic", type=int, default=80)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    md_files, md_chars, examples = generate(
        seed=args.seed,
        examples_per_category=args.examples_per_category,
        markdown_sections_per_topic=args.markdown_sections_per_topic,
        out_dir=Path(args.out_dir),
        overwrite=args.overwrite,
    )
    print(f"Generated Markdown files: {md_files}")
    print(f"Generated Markdown characters: {md_chars}")
    print(f"Generated SFT examples: {examples}")
    print(f"Output directory: {Path(args.out_dir)}")


if __name__ == "__main__":
    main()







