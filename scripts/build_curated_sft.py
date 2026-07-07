from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "assistant_sft" / "curated_personal_assistant_core.jsonl"
OUT = ROOT / "examples" / "byteseed_curated_sft.jsonl"
EXPECTED_COUNTS = {
    "dsa_study": 120,
    "ai_ml_basics": 60,
    "byteseed_workflow": 60,
    "coding_help": 60,
    "troubleshooting": 60,
    "safety_boundary": 40,
}
FORBIDDEN_ANYWHERE = [
    "Check 123",
    "Example 123",
]
FORBIDDEN_OUTSIDE_TROUBLESHOOTING = [
    "Treat it as a small debugging task",
    "Keep the change small so you can tell whether it helped",
    "send the exact error",
    "copy the full traceback",
    "check logs",
    "what changed recently",
]
FORBIDDEN_DSA = [
    "Treat it as a small debugging task",
    "Keep the change small so you can tell whether it helped",
]


def chat_text(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>"


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with RAW.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {RAW}:{line_no}: {exc}") from exc
            user = str(row.get("user", "")).strip()
            assistant = str(row.get("assistant", "")).strip()
            category = str(row.get("category", "")).strip()
            if not user or not assistant:
                raise ValueError(f"Missing user or assistant in {RAW}:{line_no}")
            if category not in EXPECTED_COUNTS:
                raise ValueError(f"Invalid or missing category in {RAW}:{line_no}: {category!r}")
            rows.append({"category": category, "user": user, "assistant": assistant})
    return rows


def duplicate_count(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def leakage_warnings(rows: list[dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    for index, row in enumerate(rows, start=1):
        category = row["category"]
        text = f"{row['user']}\n{row['assistant']}"
        lower = text.lower()
        for phrase in FORBIDDEN_ANYWHERE:
            if phrase.lower() in lower:
                warnings.append(f"line {index}: forbidden artificial label phrase {phrase!r}")
        for phrase in FORBIDDEN_OUTSIDE_TROUBLESHOOTING:
            if category != "troubleshooting" and phrase.lower() in lower:
                warnings.append(f"line {index}: troubleshooting phrase outside troubleshooting: {phrase!r}")
        if category == "dsa_study":
            for phrase in FORBIDDEN_DSA:
                if phrase.lower() in lower:
                    warnings.append(f"line {index}: DSA leakage phrase: {phrase!r}")
    return warnings


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"Missing curated raw file: {RAW}")
    rows = read_rows()
    users = [row["user"] for row in rows]
    assistants = [row["assistant"] for row in rows]
    duplicate_users = duplicate_count(users)
    duplicate_assistants = duplicate_count(assistants)
    counts = Counter(row["category"] for row in rows)
    warnings = leakage_warnings(rows)

    print(f"Raw curated file: {RAW.relative_to(ROOT)}")
    print(f"Total examples: {len(rows)}")
    print("Examples by category:")
    for category, expected in EXPECTED_COUNTS.items():
        actual = counts.get(category, 0)
        print(f"  - {category}: {actual}")
        if actual != expected:
            raise SystemExit(f"Expected {expected} examples for {category}, found {actual}")
    print(f"Duplicate user prompts: {duplicate_users}")
    print(f"Duplicate assistant responses: {duplicate_assistants}")
    if duplicate_users or duplicate_assistants:
        raise SystemExit("Duplicate curated examples found; refusing to write output.")
    if warnings:
        print(f"Forbidden leakage warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Forbidden leakage warnings: 0")

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
    print(f"Wrote curated SFT JSONL: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
