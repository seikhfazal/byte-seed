from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "assistant_sft" / "anchor_identity_dsa.jsonl"
OUT = ROOT / "examples" / "byteseed_anchor_sft.jsonl"
EXPECTED_COUNTS = {"identity": 40, "dsa_study": 40, "byteseed_workflow": 20}


def chat_text(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>"


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with RAW.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user = str(row.get("user", "")).strip()
            assistant = str(row.get("assistant", "")).strip()
            category = str(row.get("category", "")).strip()
            if not user or not assistant:
                raise ValueError(f"Missing user or assistant in {RAW}:{line_no}")
            if category not in EXPECTED_COUNTS:
                raise ValueError(f"Invalid category in {RAW}:{line_no}: {category!r}")
            rows.append({"user": user, "assistant": assistant, "category": category})
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not RAW.exists():
        raise SystemExit(f"Missing anchor raw file: {RAW}")
    rows = read_rows()
    counts = Counter(row["category"] for row in rows)
    for category, expected in EXPECTED_COUNTS.items():
        actual = counts.get(category, 0)
        if actual != expected:
            raise SystemExit(f"Expected {expected} {category} examples, found {actual}.")
    if len(rows) != sum(EXPECTED_COUNTS.values()):
        raise SystemExit(f"Expected 100 examples, found {len(rows)}.")

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
    print(f"Wrote {len(rows)} anchor SFT examples to {OUT.relative_to(ROOT)}")
    for category, count in counts.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
