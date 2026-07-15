from __future__ import annotations

import importlib

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
    root_chat = importlib.import_module("chat")
    monkeypatch.setattr(root_chat, "PREFERRED_CHECKPOINTS", ("missing-first.pt", "fallback.pt"))

    assert root_chat.default_checkpoint() == "fallback.pt"
