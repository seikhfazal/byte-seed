from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "public_imports"

SOURCES = {
    "dolly": {
        "dataset_id": "databricks/databricks-dolly-15k",
        "license": "CC-BY-SA-3.0",
    },
    "oasst1": {
        "dataset_id": "OpenAssistant/oasst1",
        "license": "Apache-2.0",
    },
    "hh-rlhf": {
        "dataset_id": "Anthropic/hh-rlhf",
        "license": "See dataset card; preference data, not direct chat SFT by default",
    },
}


def require_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The optional 'datasets' package is not installed.\n"
            "Install it with: pip install -r requirements-data.txt"
        ) from exc
    return load_dataset


def clean(text: object) -> str:
    return str(text or "").replace("\r\n", "\n").strip()


def write_jsonl(path: Path, rows: Iterable[dict], overwrite: bool) -> int:
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def convert_dolly(limit: int | None):
    load_dataset = require_datasets()
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    produced = 0
    for item in dataset:
        instruction = clean(item.get("instruction"))
        context = clean(item.get("context"))
        response = clean(item.get("response"))
        if not instruction or not response:
            continue
        user = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        row = {
            "user": user,
            "assistant": response,
            "source": "databricks/databricks-dolly-15k",
            "license": "CC-BY-SA-3.0",
        }
        category = clean(item.get("category"))
        if category:
            row["category"] = category
        yield row
        produced += 1
        if limit is not None and produced >= limit:
            break


def convert_oasst1(limit: int | None):
    load_dataset = require_datasets()
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    by_id: dict[str, dict] = {}
    for item in dataset:
        message_id = clean(item.get("message_id"))
        if message_id:
            by_id[message_id] = item

    produced = 0
    for item in dataset:
        if clean(item.get("role")) != "assistant":
            continue
        parent_id = clean(item.get("parent_id"))
        parent = by_id.get(parent_id)
        if not parent or clean(parent.get("role")) != "prompter":
            continue
        user = clean(parent.get("text"))
        assistant = clean(item.get("text"))
        if not user or not assistant:
            continue
        yield {
            "user": user,
            "assistant": assistant,
            "source": "OpenAssistant/oasst1",
            "license": "Apache-2.0",
            "note": "Best-effort parent prompt to assistant reply pair; conversation tree context is not fully preserved.",
        }
        produced += 1
        if limit is not None and produced >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Optionally import small public assistant datasets into ByteSeed JSONL format.")
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.source == "hh-rlhf":
        raise SystemExit(
            "Anthropic/hh-rlhf is preference data, not direct chat SFT by default. "
            "Add a custom converter only after reviewing the dataset card and license."
        )

    source_info = SOURCES[args.source]
    print(f"Source: {source_info['dataset_id']}")
    print(f"License reminder: {source_info['license']}")
    print("Warning: public imported datasets should not be blindly committed. Review and keep them local by default.")

    rows = convert_dolly(args.limit) if args.source == "dolly" else convert_oasst1(args.limit)
    out_path = OUT_DIR / f"{args.source}_{args.limit or 'all'}.jsonl"
    count = write_jsonl(out_path, rows, args.overwrite)
    print(f"Wrote {count} examples to {out_path}")


if __name__ == "__main__":
    main()
