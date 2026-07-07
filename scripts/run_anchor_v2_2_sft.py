from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ByteSeed on Anchor v2.2 cleanup data only.")
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    examples = ROOT / "examples" / "byteseed_anchor_v2_2_sft.jsonl"
    checkpoint = ROOT / "checkpoints" / "anchor_v2_1_finetuned.pt"
    if not examples.exists():
        raise SystemExit("Missing examples/byteseed_anchor_v2_2_sft.jsonl. Run scripts/build_anchor_v2_2_sft.py first.")
    if not checkpoint.exists():
        raise SystemExit("Missing checkpoints/anchor_v2_1_finetuned.pt. Anchor v2.2 cleanup SFT starts from Anchor v2.1.")

    command = [
        sys.executable,
        "-m",
        "src.byteseed.finetune_chat",
        "--config",
        args.config,
        "--checkpoint",
        "checkpoints/anchor_v2_1_finetuned.pt",
        "--examples",
        "examples/byteseed_anchor_v2_2_sft.jsonl",
        "--iters",
        str(args.iters),
        "--output",
        "checkpoints/anchor_v2_2_finetuned.pt",
    ]
    print("Anchor v2.2 cleanup SFT:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
