from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from byteseed import chat as chat_module


def test_preferred_checkpoint_uses_declared_priority_without_deserializing(monkeypatch, tmp_path):
    preferred = tmp_path / "anchor_v2_3_finetuned.pt"
    fallback = tmp_path / "anchor_v2_2_finetuned.pt"
    preferred.touch()
    fallback.touch()
    monkeypatch.setattr(chat_module, "PREFERRED_CHECKPOINTS", (str(preferred), str(fallback)))

    assert chat_module.preferred_checkpoint() == str(preferred)


def test_explicit_checkpoint_label_wins_without_reading_config():
    assert chat_module.resolve_checkpoint_label("not-read.yaml", "custom.pt") == "custom.pt"


def test_missing_preferred_checkpoints_return_none(monkeypatch, tmp_path):
    monkeypatch.setattr(chat_module, "PREFERRED_CHECKPOINTS", (str(tmp_path / "missing.pt"),))

    assert chat_module.preferred_checkpoint() is None


def test_root_chat_default_falls_back_to_last_documented_name(monkeypatch):
    launcher_path = Path(__file__).resolve().parents[1] / "chat.py"
    spec = importlib.util.spec_from_file_location("byteseed_root_chat_test", launcher_path)
    assert spec is not None
    assert spec.loader is not None
    root_chat = importlib.util.module_from_spec(spec)
    repository_root = str(launcher_path.parent)
    already_on_path = repository_root in sys.path
    if not already_on_path:
        sys.path.insert(0, repository_root)
    try:
        spec.loader.exec_module(root_chat)
    finally:
        if not already_on_path:
            sys.path.remove(repository_root)
    monkeypatch.setattr(root_chat, "PREFERRED_CHECKPOINTS", ("missing-first.pt", "fallback.pt"))

    assert root_chat.default_checkpoint() == "fallback.pt"
