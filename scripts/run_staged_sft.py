from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_stage(name: str, checkpoint: str, examples: str, output: str, iters: int, config: str) -> None:
    command = [
        sys.executable,
        "-m",
        "src.byteseed.finetune_chat",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--examples",
        examples,
        "--iters",
        str(iters),
        "--output",
        output,
    ]
    print(f"{name}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged ByteSeed SFT: anchor first, curated second.")
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--stage1-iters", type=int, default=600)
    parser.add_argument("--stage2-iters", type=int, default=300)
    parser.add_argument("--skip-stage2", action="store_true")
    args = parser.parse_args()

    anchor_examples = "examples/byteseed_anchor_sft.jsonl"
    curated_examples = "examples/byteseed_curated_sft.jsonl"
    if not (ROOT / anchor_examples).exists():
        raise SystemExit(f"Missing {anchor_examples}. Run scripts/build_anchor_sft.py first.")
    if not (ROOT / "checkpoints" / "best.pt").exists():
        raise SystemExit("Missing checkpoints/best.pt. Stage 1 needs the pretrained checkpoint.")

    run_stage(
        "Stage 1 anchor SFT",
        "checkpoints/best.pt",
        anchor_examples,
        "checkpoints/anchor_finetuned.pt",
        args.stage1_iters,
        args.config,
    )
    if args.skip_stage2:
        print("Skipping Stage 2.")
        return
    if not (ROOT / curated_examples).exists():
        raise SystemExit(f"Missing {curated_examples}.")
    run_stage(
        "Stage 2 light curated SFT",
        "checkpoints/anchor_finetuned.pt",
        curated_examples,
        "checkpoints/chat_finetuned.pt",
        args.stage2_iters,
        args.config,
    )


if __name__ == "__main__":
    main()

