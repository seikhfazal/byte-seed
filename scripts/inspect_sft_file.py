from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IDENTITY_PATTERNS = ("who are you", "what are you", "tell me about yourself")
DSA_PATTERNS = ("dsa", "study session", "linked list", "stack", "queue")
SAFETY_PATTERNS = (
    "safety",
    "boundary",
    "private data",
    "redact",
    "secret",
    "password",
    "token",
)


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            rows.append(row)
    return rows


def duplicate_items(values: list[str]) -> list[tuple[str, int]]:
    return [(value, count) for value, count in Counter(values).most_common() if count > 1]


def print_examples(title: str, rows: list[dict[str, Any]], limit: int = 10) -> None:
    print(title)
    if not rows:
        print("  none")
        return
    for index, row in enumerate(rows[:limit], start=1):
        category = row.get("category", "")
        category_text = f" [{category}]" if category else ""
        print(f"  {index}.{category_text} user: {str(row.get('user', '')).strip()}")
        print(f"     assistant: {str(row.get('assistant', '')).strip()}")
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Inspect a ByteSeed SFT JSONL file.")
    parser.add_argument("--path", default="examples/byteseed_curated_sft.jsonl")
    args = parser.parse_args()

    path = Path(args.path)
    rows = read_rows(path)
    users = [str(row.get("user", "")).strip() for row in rows]
    assistants = [str(row.get("assistant", "")).strip() for row in rows]
    categories = Counter(str(row.get("category", "")).strip() for row in rows if row.get("category"))
    user_lower = [user.lower() for user in users]
    assistant_lower = [assistant.lower() for assistant in assistants]

    identity_rows = [row for row, user in zip(rows, user_lower) if any(pattern in user for pattern in IDENTITY_PATTERNS)]
    dsa_rows = [row for row, user in zip(rows, user_lower) if any(pattern in user for pattern in DSA_PATTERNS)]
    if categories:
        safety_rows = [
            row
            for row in rows
            if str(row.get("category", "")).strip().lower() in {"safety", "safety_boundary", "boundary"}
        ]
    else:
        safety_rows = [
            row
            for row, user, assistant in zip(rows, user_lower, assistant_lower)
            if any(pattern in f"{user}\n{assistant}" for pattern in SAFETY_PATTERNS)
        ]
    no_answers = [row for row, assistant in zip(rows, assistants) if assistant.startswith("No.")]
    user_dupes = duplicate_items(users)
    assistant_dupes = duplicate_items(assistants)
    first_8 = Counter(" ".join(normalize_words(assistant)[:8]) for assistant in assistants if normalize_words(assistant))

    print(f"path: {path}")
    print(f"total examples: {len(rows)}")
    if categories:
        print("categories:")
        for category, count in categories.most_common():
            print(f"  {category}: {count}")
    else:
        print("categories: none")
    print()

    print_examples("first 5 examples:", rows, limit=5)
    print()
    print_examples("identity prompt matches:", identity_rows, limit=20)
    print()
    print_examples("DSA study/planning matches:", dsa_rows, limit=20)
    print()

    print(f"safety/boundary example count: {len(safety_rows)}")
    print(f'assistant answers starting with "No.": {len(no_answers)}')
    print(f"duplicate user prompts: {sum(count - 1 for _, count in user_dupes)}")
    for value, count in user_dupes[:20]:
        print(f"  {count}x user: {value}")
    print(f"duplicate assistant responses: {sum(count - 1 for _, count in assistant_dupes)}")
    for value, count in assistant_dupes[:20]:
        print(f"  {count}x assistant: {value}")
    print("top 20 assistant first 8 words:")
    for phrase, count in first_8.most_common(20):
        print(f"  {count}x {phrase}")


if __name__ == "__main__":
    main()


