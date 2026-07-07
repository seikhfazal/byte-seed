from __future__ import annotations

from pathlib import Path

from src.byteseed.chat import main


def default_checkpoint() -> str:
    for checkpoint in (
        "checkpoints/anchor_v2_2_finetuned.pt",
        "checkpoints/anchor_v2_1_finetuned.pt",
        "checkpoints/anchor_v2_finetuned.pt",
        "checkpoints/anchor_finetuned.pt",
        "checkpoints/chat_finetuned.pt",
    ):
        if Path(checkpoint).exists():
            return checkpoint
    return "checkpoints/chat_finetuned.pt"


if __name__ == "__main__":
    main(
        default_config="configs/byteseed_12m.yaml",
        default_checkpoint=default_checkpoint(),
        default_temperature=0.3,
        default_top_k=8,
        default_max_new_tokens=120,
    )
