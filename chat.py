from __future__ import annotations

from pathlib import Path

from src.byteseed.chat import main


PREFERRED_CHECKPOINTS = (
    "checkpoints/anchor_v2_3_finetuned.pt",
    "checkpoints/anchor_v2_2_finetuned.pt",
    "checkpoints/anchor_v2_1_finetuned.pt",
    "checkpoints/anchor_v2_finetuned.pt",
    "checkpoints/anchor_finetuned.pt",
    "checkpoints/chat_finetuned.pt",
)


def default_checkpoint() -> str:
    for checkpoint in PREFERRED_CHECKPOINTS:
        if Path(checkpoint).exists():
            return checkpoint
    return PREFERRED_CHECKPOINTS[-1]


if __name__ == "__main__":
    main(
        default_config="configs/byteseed_12m.yaml",
        default_checkpoint=default_checkpoint(),
        default_preset="precise",
    )
