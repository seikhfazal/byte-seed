from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSONAL_DIR = ROOT / "data" / "raw" / "personal_assistant"
SFT_DIR = ROOT / "data" / "raw" / "assistant_sft"
GENERATED_MD_DIR = ROOT / "data" / "raw" / "generated" / "markdown"
GENERATED_SFT_DIR = ROOT / "data" / "raw" / "generated" / "sft"
PUBLIC_DIR = ROOT / "data" / "public_imports"
CORPUS_OUT = ROOT / "data" / "raw" / "byteseed_personal_assistant_corpus.md"
SFT_OUT = ROOT / "examples" / "byteseed_personal_assistant_sft.jsonl"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            user = str(row.get("user", "")).strip()
            assistant = str(row.get("assistant", "")).strip()
            if user and assistant:
                yield row, user, assistant


def chat_text(user: str, assistant: str) -> str:
    return f"<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>"


def count_examples(paths: list[Path]) -> int:
    return sum(1 for path in paths for _row in read_jsonl(path))


def build(include_public: bool = False) -> tuple[int, int]:
    personal_md_files = sorted(PERSONAL_DIR.glob("*.md"))
    generated_md_files = sorted(GENERATED_MD_DIR.glob("*.md"))
    md_files = personal_md_files + generated_md_files
    if not md_files:
        raise SystemExit(f"No Markdown files found in {PERSONAL_DIR} or {GENERATED_MD_DIR}")

    parts = ["# ByteSeed Personal Assistant Corpus", ""]
    for path in md_files:
        rel = path.relative_to(ROOT).as_posix()
        parts.append(f"\n---\n\n## Source: {rel}\n")
        parts.append(path.read_text(encoding="utf-8").strip())
        parts.append("")
    corpus_text = "\n".join(parts).strip() + "\n"
    CORPUS_OUT.write_text(corpus_text, encoding="utf-8")

    handwritten_jsonl_files = sorted(SFT_DIR.glob("*.jsonl"))
    generated_jsonl_files = sorted(GENERATED_SFT_DIR.glob("*.jsonl"))
    jsonl_files = handwritten_jsonl_files + generated_jsonl_files
    if include_public:
        jsonl_files.extend(sorted(PUBLIC_DIR.glob("*.jsonl")))

    handwritten_count = count_examples(handwritten_jsonl_files)
    generated_count = count_examples(generated_jsonl_files)

    count = 0
    SFT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SFT_OUT.open("w", encoding="utf-8") as out:
        for path in jsonl_files:
            for row, user, assistant in read_jsonl(path):
                source = row.get("source", path.relative_to(ROOT).as_posix())
                out_row = {
                    "user": user,
                    "assistant": assistant,
                    "text": chat_text(user, assistant),
                    "source": source,
                }
                if "license" in row:
                    out_row["license"] = row["license"]
                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote Markdown corpus: {CORPUS_OUT}")
    print(f"Wrote SFT JSONL: {SFT_OUT}")
    print(f"Personal handwritten markdown files: {len(personal_md_files)}")
    print(f"Generated markdown files: {len(generated_md_files)}")
    print(f"Handwritten SFT examples: {handwritten_count}")
    print(f"Generated SFT examples: {generated_count}")
    print(f"Total SFT examples: {count}")
    print(f"Total Markdown characters: {len(corpus_text)}")
    if include_public:
        print("Included public imports. Review licenses and do not commit imported datasets blindly.")
    return len(md_files), count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ByteSeed personal assistant corpus and SFT examples.")
    parser.add_argument("--include-public", action="store_true", help="Also include data/public_imports/*.jsonl")
    args = parser.parse_args()
    build(include_public=args.include_public)


if __name__ == "__main__":
    main()
