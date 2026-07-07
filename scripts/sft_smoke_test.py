from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.byteseed.config import load_config
from src.byteseed.finetune_chat import finetune, generate_reply
from src.byteseed.generate import load_model
from src.byteseed.tokenizer import ByteSeedTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny controlled SFT smoke test.")
    parser.add_argument("--config", default="configs/byteseed_12m.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--examples", default="data/raw/assistant_sft/identity_test.jsonl")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--output", default="checkpoints/sft_smoke.pt")
    args = parser.parse_args()

    checkpoint = args.checkpoint if Path(args.checkpoint).exists() else None
    if checkpoint is None:
        print(f"Warning: {args.checkpoint} not found; using latest configured checkpoint.")

    out = finetune(args.config, checkpoint, args.examples, args.iters, output=args.output, mask_prompt=True)
    cfg = load_config(args.config)
    tokenizer = ByteSeedTokenizer(cfg.tokenizer_dir)
    model = load_model(cfg, str(out))
    model.eval()

    prompts = [
        "<|user|>\nwho are you?\n<|assistant|>\n",
        "<|user|>\nHelp me plan a 1 hour DSA study session.\n<|assistant|>\n",
    ]
    with torch.no_grad():
        for prompt in prompts:
            print("PROMPT:")
            print(prompt, end="")
            print("OUTPUT:")
            print(generate_reply(model, tokenizer, prompt))
            print("---")


if __name__ == "__main__":
    main()
