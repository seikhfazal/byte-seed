from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CORPUS_OUT = RAW_DIR / "byteseed_personal_assistant_corpus.md"
SFT_DIR = RAW_DIR / "assistant_sft"
GENERATED_SFT_DIR = RAW_DIR / "generated" / "sft"
PUBLIC_DIR = ROOT / "data" / "public_imports"
TOKENIZER_META_JSON = ROOT / "tokenizer" / "tokenizer_meta.json"
TOKENIZER_META_YAML = ROOT / "tokenizer" / "tokenizer_meta.yaml"
SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"hf_[A-Za-z0-9]{20,}",
    r"(?i)api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}",
    r"(?i)password\s*[:=]\s*['\"]?[^\s'\"]{8,}",
    r"(?i)token\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}",
]
GITHUB_HYGIENE_PHRASES = [
    "keep the public repo clean",
    "commit source",
    "do not commit checkpoints",
    "public github repo",
    "commit message",
    "gitignore",
]
ARTIFICIAL_CHECK_LABEL_RE = re.compile(r"\bCheck\s+\d+\b")
ARTIFICIAL_EXAMPLE_LABEL_RE = re.compile(r"\bExample\s+\d+\b")
ORDINARY_CHECK_RE = re.compile(r"\bcheck\b", re.IGNORECASE)


def iter_text_files(include_public: bool):
    for path in RAW_DIR.rglob("*.md"):
        yield path
    for path in SFT_DIR.glob("*.jsonl"):
        yield path
    for path in GENERATED_SFT_DIR.glob("*.jsonl"):
        yield path
    if include_public:
        for path in PUBLIC_DIR.glob("*.jsonl"):
            yield path


def inspect_jsonl(path: Path):
    count = 0
    shortest = None
    longest = 0
    users: list[str] = []
    assistants: list[str] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: invalid JSON in {path}:{line_no}: {exc}")
                continue
            user = str(row.get("user", "")).strip()
            assistant = str(row.get("assistant", "")).strip()
            text = f"{user}\n{assistant}"
            length = len(text)
            shortest = length if shortest is None else min(shortest, length)
            longest = max(longest, length)
            users.append(user)
            assistants.append(assistant)
            count += 1
    return count, shortest or 0, longest, users, assistants


def duplicate_count(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def tokenizer_meta_missing_specials() -> bool:
    if TOKENIZER_META_JSON.exists():
        try:
            meta = json.loads(TOKENIZER_META_JSON.read_text(encoding="utf-8"))
            symbols = set(meta.get("user_defined_symbols", []))
            return not {"<|user|>", "<|assistant|>"}.issubset(symbols)
        except json.JSONDecodeError:
            return True
    if TOKENIZER_META_YAML.exists():
        text = TOKENIZER_META_YAML.read_text(encoding="utf-8", errors="replace")
        return "<|user|>" not in text or "<|assistant|>" not in text
    return True


def print_top_repeats(title: str, values: list[str], limit: int = 5) -> None:
    repeated = [(value, count) for value, count in Counter(values).most_common(limit) if count > 1]
    print(title)
    if not repeated:
        print("  - none")
        return
    for value, count in repeated:
        clean = value.replace("\n", " ")[:120]
        print(f"  - {count}x: {clean}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ByteSeed local dataset size and possible quality issues.")
    parser.add_argument("--include-public", action="store_true", help="Also inspect data/public_imports/*.jsonl")
    args = parser.parse_args()

    md_files = sorted(path for path in RAW_DIR.rglob("*.md") if path != CORPUS_OUT)
    total_chars = sum(len(path.read_text(encoding="utf-8-sig", errors="replace")) for path in md_files)
    rough_tokens = total_chars // 4

    jsonl_files = sorted(SFT_DIR.glob("*.jsonl")) + sorted(GENERATED_SFT_DIR.glob("*.jsonl"))
    if args.include_public:
        jsonl_files.extend(sorted(PUBLIC_DIR.glob("*.jsonl")))

    total_examples = 0
    shortest = None
    longest = 0
    all_users: list[str] = []
    all_assistants: list[str] = []
    examples_by_file: list[tuple[Path, int]] = []
    github_leaks: list[str] = []
    for path in jsonl_files:
        count, short, long, users, assistants = inspect_jsonl(path)
        examples_by_file.append((path, count))
        total_examples += count
        all_users.extend(users)
        all_assistants.extend(assistants)
        if count:
            shortest = short if shortest is None else min(shortest, short)
            longest = max(longest, long)
        if path.parent == GENERATED_SFT_DIR and path.name != "generated_github_hygiene.jsonl":
            text = "\n".join(users + assistants).lower()
            for phrase in GITHUB_HYGIENE_PHRASES:
                hits = text.count(phrase)
                if hits:
                    github_leaks.append(f"{path.relative_to(ROOT)} has {hits} occurrence(s) of '{phrase}'")

    suspicious: list[str] = []
    data_has_chat_markers = False
    all_text_parts: list[str] = []
    for path in iter_text_files(args.include_public):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        all_text_parts.append(text)
        if "<|user|>" in text or "<|assistant|>" in text:
            data_has_chat_markers = True
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text):
                suspicious.append(f"{path.relative_to(ROOT)} matches {pattern}")
    all_text = "\n".join(all_text_parts)
    artificial_check_label_count = len(ARTIFICIAL_CHECK_LABEL_RE.findall(all_text))
    artificial_example_label_count = len(ARTIFICIAL_EXAMPLE_LABEL_RE.findall(all_text))
    ordinary_check_count = len(ORDINARY_CHECK_RE.findall(all_text))
    first80 = [assistant[:80] for assistant in all_assistants]
    most_common_prefix = Counter(first80).most_common(1)

    print(f"Total .md files: {len(md_files)}")
    print(f"Total Markdown characters: {total_chars}")
    print(f"Rough token estimate: {rough_tokens}")
    print(f"Total JSONL examples: {total_examples}")
    print("Examples by file:")
    for path, count in examples_by_file:
        print(f"  - {path.relative_to(ROOT)}: {count}")
    print(f"Duplicate user prompts count: {duplicate_count(all_users)}")
    print(f"Duplicate assistant responses count: {duplicate_count(all_assistants)}")
    print(f"Artificial Check-label count: {artificial_check_label_count}")
    print(f"Artificial Example-label count: {artificial_example_label_count}")
    print(f"Ordinary/any-case 'check' count: {ordinary_check_count}")
    print(f"Shortest example length: {shortest or 0}")
    print(f"Longest example length: {longest}")
    print_top_repeats("Top repeated assistant responses:", all_assistants)
    print_top_repeats("Top repeated first 80 characters of assistant responses:", first80)

    if total_chars < 100_000:
        print("Tokenizer target suggestion: warning, Markdown corpus is very tiny.")
    elif total_chars <= 500_000:
        print("Tokenizer target suggestion: okay for a toy model.")
    else:
        print("Tokenizer target suggestion: better for ByteSeed.")

    if total_examples < 100:
        print("Warning: SFT dataset is tiny; assistant behavior will be limited.")
    if most_common_prefix and most_common_prefix[0][1] > max(25, total_examples // 20):
        print(f"Warning: one assistant opening pattern appears too often: {most_common_prefix[0][1]} times.")
    if github_leaks:
        print("Warning: GitHub hygiene phrases appear outside generated GitHub category files:")
        for item in github_leaks:
            print(f"  - {item}")
    if data_has_chat_markers and tokenizer_meta_missing_specials():
        print("Warning: <|user|> or <|assistant|> appears in data, but tokenizer special-token metadata is missing or incomplete.")
    if suspicious:
        print("Warning: suspicious secret-like patterns found:")
        for item in suspicious:
            print(f"  - {item}")
        print("Review these manually. Some may be educational placeholder warnings, not real secrets.")
    else:
        print("No suspicious secret-like patterns found.")


if __name__ == "__main__":
    main()





