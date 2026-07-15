from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from byteseed.provenance import (
    DATA_MANIFEST_VERSION,
    REQUIRED_SPECIAL_TOKENS,
    ProvenanceMismatchError,
    ProvenanceValidationError,
    build_pretraining_data_manifest,
    canonical_sha256,
    compare_data_manifests,
    compare_tokenizer_identities,
    create_data_manifest,
    create_tokenizer_identity,
    normalize_logical_name,
    sha256_file,
    validate_data_manifest,
    validate_tokenizer_identity,
)


def _special_tokens(offset: int = 0) -> dict[str, int]:
    return {
        token: index + offset
        for index, token in enumerate(REQUIRED_SPECIAL_TOKENS)
    }


def _tokenizer_identity(directory: Path, content: bytes = b"tokenizer-a"):
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "byteseed.model"
    model.write_bytes(content)
    return create_tokenizer_identity(
        model,
        vocab_size=32,
        special_tokens=_special_tokens(),
    )


def _artifacts(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "train.npy", np.arange(24, dtype=np.uint16))
    np.save(directory / "val.npy", np.arange(8, dtype=np.uint16))


def test_streaming_sha256_matches_known_content(tmp_path):
    path = tmp_path / "input.bin"
    content = b"ByteSeed provenance\x00\xff"
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_file_fingerprint_depends_on_bytes_not_path_or_metadata(tmp_path):
    first = tmp_path / "a" / "input.bin"
    second = tmp_path / "b" / "renamed.bin"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")
    first.touch()

    assert sha256_file(first) == sha256_file(second)

    second.write_bytes(b"same byteS")
    assert sha256_file(first) != sha256_file(second)


def test_streaming_hash_uses_bounded_reads(monkeypatch, tmp_path):
    path = tmp_path / "large.bin"
    path.write_bytes(b"x" * 101)
    original_open = Path.open
    read_sizes: list[int] = []

    class RecordingReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def read(self, size: int = -1):
            read_sizes.append(size)
            return self.handle.read(size)

    def recording_open(file_path: Path, *args, **kwargs):
        return RecordingReader(original_open(file_path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", recording_open)

    sha256_file(path, chunk_size=7)

    assert read_sizes
    assert set(read_sizes) == {7}


def test_hash_missing_file_and_directory_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        sha256_file(tmp_path / "missing.bin")
    with pytest.raises(IsADirectoryError, match="must be a file"):
        sha256_file(tmp_path)


def test_canonical_json_digest_ignores_dictionary_insertion_order():
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_logical_paths_normalize_windows_and_posix_separators():
    assert normalize_logical_name(r"shards\train.npy") == "shards/train.npy"
    assert normalize_logical_name("shards/train.npy") == "shards/train.npy"


def test_identical_tokenizer_bytes_in_different_directories_match(tmp_path):
    first = _tokenizer_identity(tmp_path / "first")
    second = _tokenizer_identity(tmp_path / "second")

    compare_tokenizer_identities(first, second)
    assert first["digest"] == second["digest"]


def test_same_sized_different_tokenizer_bytes_do_not_match(tmp_path):
    first = _tokenizer_identity(tmp_path / "first", b"tokenizer-a")
    second = _tokenizer_identity(tmp_path / "second", b"tokenizer-b")

    assert first["vocab_size"] == second["vocab_size"]
    with pytest.raises(ProvenanceMismatchError, match="tokenizer.model bytes"):
        compare_tokenizer_identities(first, second)


def test_changed_special_token_mapping_does_not_match(tmp_path):
    model = tmp_path / "byteseed.model"
    model.write_bytes(b"same tokenizer bytes")
    first = create_tokenizer_identity(model, vocab_size=32, special_tokens=_special_tokens())
    changed = _special_tokens()
    changed["<|assistant|>"] = 31
    second = create_tokenizer_identity(model, vocab_size=32, special_tokens=changed)

    with pytest.raises(ProvenanceMismatchError, match="special-token IDs"):
        compare_tokenizer_identities(first, second)


def test_tokenizer_identity_rejects_future_version_and_algorithm(tmp_path):
    identity = _tokenizer_identity(tmp_path)
    future = dict(identity, version=2)
    with pytest.raises(ProvenanceValidationError, match="Unsupported tokenizer identity version"):
        validate_tokenizer_identity(future)

    unsupported = dict(identity, algorithm="sha512")
    with pytest.raises(ProvenanceValidationError, match="hash algorithm"):
        validate_tokenizer_identity(unsupported)


def test_tokenizer_identity_missing_model_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="tokenizer.model file is missing"):
        create_tokenizer_identity(
            tmp_path / "missing.model",
            vocab_size=32,
            special_tokens=_special_tokens(),
        )


def test_pretraining_manifest_describes_consumed_train_and_validation_arrays(
    tmp_path, tokenizer_identity
):
    _artifacts(tmp_path)

    manifest = build_pretraining_data_manifest(
        tmp_path,
        tokenizer_identity=tokenizer_identity,
        train_split=0.75,
    )

    assert manifest["version"] == DATA_MANIFEST_VERSION
    assert manifest["tokenizer"]["digest"] == tokenizer_identity["digest"]
    assert [item["role"] for item in manifest["artifacts"]] == [
        "training corpus",
        "validation corpus",
    ]
    assert [item["token_count"] for item in manifest["artifacts"]] == [24, 8]
    assert all(item["format"] == "numpy-npy" for item in manifest["artifacts"])
    assert manifest["preprocessing"]["split"]["train_fraction"] == 0.75


def test_corpus_and_split_changes_change_manifest_identity(tmp_path, tokenizer_identity):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _artifacts(first_dir)
    _artifacts(second_dir)
    first = build_pretraining_data_manifest(
        first_dir, tokenizer_identity=tokenizer_identity, train_split=0.75
    )
    train = np.load(second_dir / "train.npy", allow_pickle=False)
    train[0] = 99
    np.save(second_dir / "train.npy", train)
    changed_corpus = build_pretraining_data_manifest(
        second_dir, tokenizer_identity=tokenizer_identity, train_split=0.75
    )
    changed_split = build_pretraining_data_manifest(
        first_dir, tokenizer_identity=tokenizer_identity, train_split=0.8
    )

    with pytest.raises(ProvenanceMismatchError, match="training corpus"):
        compare_data_manifests(first, changed_corpus)
    with pytest.raises(ProvenanceMismatchError, match="split configuration"):
        compare_data_manifests(first, changed_split)


def test_artifact_order_and_non_identity_metadata_do_not_change_digest(data_manifest):
    reversed_artifacts = list(reversed(data_manifest["artifacts"]))
    first = create_data_manifest(
        tokenizer_identity=data_manifest["tokenizer"],
        artifacts=reversed_artifacts,
        preprocessing=data_manifest["preprocessing"],
        metadata={"created_at": "first volatile value"},
    )
    second = create_data_manifest(
        tokenizer_identity=data_manifest["tokenizer"],
        artifacts=data_manifest["artifacts"],
        preprocessing=data_manifest["preprocessing"],
        metadata={"created_at": "second volatile value"},
    )

    assert first["digest"] == second["digest"]


def test_manifest_missing_artifact_and_malformed_metadata_fail(tmp_path, tokenizer_identity):
    _artifacts(tmp_path)
    (tmp_path / "val.npy").unlink()
    with pytest.raises(FileNotFoundError, match="validation corpus file is missing"):
        build_pretraining_data_manifest(
            tmp_path, tokenizer_identity=tokenizer_identity, train_split=0.75
        )

    malformed = {
        "version": DATA_MANIFEST_VERSION,
        "algorithm": "sha256",
        "tokenizer": tokenizer_identity,
        "artifacts": [{"role": "training corpus"}],
        "preprocessing": {
            "version": 1,
            "split": {"strategy": "contiguous-token-fraction", "train_fraction": 0.75},
        },
        "digest": "0" * 64,
    }
    with pytest.raises(ProvenanceValidationError, match="[Ll]ogical artifact name"):
        validate_data_manifest(malformed)


def test_future_data_manifest_version_fails_clearly(data_manifest):
    future = dict(data_manifest, version=DATA_MANIFEST_VERSION + 1)
    with pytest.raises(ProvenanceValidationError, match="Unsupported data manifest version"):
        validate_data_manifest(future)

    unsupported = dict(data_manifest, algorithm="sha512")
    with pytest.raises(ProvenanceValidationError, match="hash algorithm"):
        validate_data_manifest(unsupported)
