from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


HASH_ALGORITHM = "sha256"
TOKENIZER_IDENTITY_VERSION = 1
DATA_MANIFEST_VERSION = 1
CHECKPOINT_PROVENANCE_VERSION = 1
DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024

REQUIRED_SPECIAL_TOKENS = (
    "<unk>",
    "<s>",
    "</s>",
    "<pad>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceError(RuntimeError):
    """Base error for ByteSeed identity and manifest failures."""


class ProvenanceValidationError(ProvenanceError):
    """Raised when an identity or manifest is malformed or unsupported."""


class ProvenanceMismatchError(ProvenanceError):
    """Raised when two valid provenance records identify different inputs."""

    def __init__(self, component: str, message: str):
        self.component = component
        super().__init__(message)


def sha256_file(path: str | Path, *, chunk_size: int = DEFAULT_HASH_CHUNK_SIZE) -> str:
    """Return the lowercase SHA-256 of file bytes using bounded reads."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Fingerprint input file is missing: {file_path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Fingerprint input must be a file: {file_path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise ProvenanceError(f"Could not read fingerprint input {file_path}: {exc}") from exc
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize identity data with stable UTF-8 JSON settings."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProvenanceValidationError(
            f"Identity data is not canonically JSON-serializable: {exc}"
        ) from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalize_logical_name(value: str | Path) -> str:
    """Normalize a portable relative artifact name without machine-specific roots."""
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or re.match(r"^[A-Za-z]:", raw):
        raise ProvenanceValidationError(
            f"Logical artifact name must be a non-empty relative path: {value!r}."
        )
    parts = [part for part in path.parts if part not in {"", "."}]
    if not parts or ".." in parts:
        raise ProvenanceValidationError(
            f"Logical artifact name cannot escape its manifest root: {value!r}."
        )
    return "/".join(parts)


def create_tokenizer_identity(
    model_path: str | Path,
    *,
    vocab_size: int,
    special_tokens: Mapping[str, int],
) -> dict[str, Any]:
    """Build a path-independent identity for the authoritative SentencePiece model."""
    model_file = Path(model_path)
    identity: dict[str, Any] = {
        "version": TOKENIZER_IDENTITY_VERSION,
        "algorithm": HASH_ALGORITHM,
        "model": {
            "logical_name": "byteseed.model",
            "size_bytes": _file_size(model_file, "tokenizer.model"),
            "sha256": sha256_file(model_file),
        },
        "vocab_size": _non_negative_int(vocab_size, "tokenizer vocab_size", positive=True),
        "special_tokens": _validated_special_tokens(special_tokens),
    }
    identity["digest"] = canonical_sha256(_tokenizer_digest_payload(identity))
    validate_tokenizer_identity(identity)
    return identity


def validate_tokenizer_identity(identity: Mapping[str, Any]) -> None:
    if not isinstance(identity, Mapping):
        raise ProvenanceValidationError("Tokenizer identity must be a mapping.")
    _validate_version(identity, "tokenizer identity", TOKENIZER_IDENTITY_VERSION)
    _validate_algorithm(identity, "tokenizer identity")
    model = identity.get("model")
    if not isinstance(model, Mapping):
        raise ProvenanceValidationError("Tokenizer identity model record must be a mapping.")
    if normalize_logical_name(model.get("logical_name", "")) != "byteseed.model":
        raise ProvenanceValidationError(
            "Tokenizer identity model logical_name must be 'byteseed.model'."
        )
    _non_negative_int(model.get("size_bytes"), "tokenizer.model size_bytes")
    _validate_digest(model.get("sha256"), "tokenizer.model sha256")
    _non_negative_int(identity.get("vocab_size"), "tokenizer vocab_size", positive=True)
    _validated_special_tokens(identity.get("special_tokens"))
    stored_digest = _validate_digest(identity.get("digest"), "tokenizer identity digest")
    expected_digest = canonical_sha256(_tokenizer_digest_payload(identity))
    if stored_digest != expected_digest:
        raise ProvenanceValidationError(
            "Tokenizer identity digest does not match its canonical fields."
        )


def compare_tokenizer_identities(
    checkpoint_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> None:
    validate_tokenizer_identity(checkpoint_identity)
    validate_tokenizer_identity(runtime_identity)
    if checkpoint_identity["digest"] == runtime_identity["digest"]:
        return

    differences: list[str] = []
    checkpoint_model = checkpoint_identity["model"]
    runtime_model = runtime_identity["model"]
    if checkpoint_model["sha256"] != runtime_model["sha256"]:
        differences.append("tokenizer.model bytes")
    if checkpoint_model["size_bytes"] != runtime_model["size_bytes"]:
        differences.append("tokenizer.model size")
    if checkpoint_identity["vocab_size"] != runtime_identity["vocab_size"]:
        differences.append("vocabulary size")
    if checkpoint_identity["special_tokens"] != runtime_identity["special_tokens"]:
        differences.append("special-token IDs")
    detail = ", ".join(differences) or "canonical tokenizer identity"
    raise ProvenanceMismatchError(
        "tokenizer",
        "Tokenizer identity mismatch in "
        f"{detail}: checkpoint={short_digest(checkpoint_identity['digest'])}, "
        f"runtime={short_digest(runtime_identity['digest'])}.",
    )


def tokenizer_identity_from_processor(
    model_path: str | Path,
    processor: Any,
) -> dict[str, Any]:
    def required_id(piece: str, fallback: int) -> int:
        token_id = int(fallback)
        if token_id < 0 or str(processor.id_to_piece(token_id)) != piece:
            raise ProvenanceValidationError(
                f"Required tokenizer special token {piece!r} is missing or mapped incorrectly."
            )
        return token_id

    special_tokens = {
        "<unk>": required_id("<unk>", processor.unk_id()),
        "<s>": required_id("<s>", processor.bos_id()),
        "</s>": required_id("</s>", processor.eos_id()),
        "<pad>": required_id("<pad>", processor.pad_id()),
    }
    for piece in REQUIRED_SPECIAL_TOKENS[4:]:
        special_tokens[piece] = required_id(piece, processor.piece_to_id(piece))
    return create_tokenizer_identity(
        model_path,
        vocab_size=int(processor.get_piece_size()),
        special_tokens=special_tokens,
    )


def fingerprint_numpy_artifact(
    path: str | Path,
    *,
    role: str,
    logical_name: str | Path | None = None,
) -> dict[str, Any]:
    artifact_path = Path(path)
    logical = normalize_logical_name(logical_name or artifact_path.name)
    size_bytes = _file_size(artifact_path, str(role))
    try:
        array = np.load(artifact_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProvenanceValidationError(
            f"Could not inspect {role} corpus artifact {artifact_path}: {exc}"
        ) from exc
    try:
        if array.ndim != 1:
            raise ProvenanceValidationError(
                f"{role} corpus artifact must be one-dimensional, got shape {array.shape}."
            )
        token_count = int(array.size)
        dtype = str(array.dtype)
    finally:
        del array
    return {
        "role": str(role),
        "logical_name": logical,
        "format": "numpy-npy",
        "size_bytes": size_bytes,
        "sha256": sha256_file(artifact_path),
        "token_count": token_count,
        "dtype": dtype,
    }


def create_data_manifest(
    *,
    tokenizer_identity: Mapping[str, Any],
    artifacts: Iterable[Mapping[str, Any]],
    preprocessing: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_tokenizer_identity(tokenizer_identity)
    normalized_artifacts = sorted(
        (_normalized_artifact(artifact) for artifact in artifacts),
        key=lambda item: (item["role"], item["logical_name"]),
    )
    if not normalized_artifacts:
        raise ProvenanceValidationError("Data manifest must contain at least one artifact.")
    manifest: dict[str, Any] = {
        "version": DATA_MANIFEST_VERSION,
        "algorithm": HASH_ALGORITHM,
        "tokenizer": dict(tokenizer_identity),
        "artifacts": normalized_artifacts,
        "preprocessing": dict(preprocessing),
    }
    if metadata is not None:
        manifest["metadata"] = dict(metadata)
    manifest["digest"] = canonical_sha256(_data_manifest_digest_payload(manifest))
    validate_data_manifest(manifest)
    return manifest


def build_pretraining_data_manifest(
    processed_data_dir: str | Path,
    *,
    tokenizer_identity: Mapping[str, Any],
    train_split: float,
) -> dict[str, Any]:
    processed = Path(processed_data_dir)
    artifacts = [
        fingerprint_numpy_artifact(
            processed / "train.npy",
            role="training corpus",
            logical_name="train.npy",
        ),
        fingerprint_numpy_artifact(
            processed / "val.npy",
            role="validation corpus",
            logical_name="val.npy",
        ),
    ]
    preprocessing = {
        "version": 1,
        "builder": "byteseed.prepare_data",
        "tokenization": {"add_bos": True, "add_eos": True},
        "split": {
            "strategy": "contiguous-token-fraction",
            "train_fraction": float(train_split),
        },
    }
    return create_data_manifest(
        tokenizer_identity=tokenizer_identity,
        artifacts=artifacts,
        preprocessing=preprocessing,
    )


def validate_data_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ProvenanceValidationError("Data manifest must be a mapping.")
    _validate_version(manifest, "data manifest", DATA_MANIFEST_VERSION)
    _validate_algorithm(manifest, "data manifest")
    validate_tokenizer_identity(manifest.get("tokenizer"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ProvenanceValidationError("Data manifest artifacts must be a non-empty list.")
    normalized = [_normalized_artifact(artifact) for artifact in artifacts]
    keys = [(item["role"], item["logical_name"]) for item in normalized]
    if len(keys) != len(set(keys)):
        raise ProvenanceValidationError("Data manifest contains duplicate artifact roles/names.")
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ProvenanceValidationError("Data manifest preprocessing must be a mapping.")
    if preprocessing.get("version") != 1:
        raise ProvenanceValidationError(
            f"Unsupported preprocessing identity version {preprocessing.get('version')!r}."
        )
    builder = preprocessing.get("builder")
    tokenization = preprocessing.get("tokenization")
    split = preprocessing.get("split")
    if not isinstance(builder, str) or not builder:
        raise ProvenanceValidationError(
            "Data manifest preprocessing builder must be non-empty text."
        )
    if not isinstance(tokenization, Mapping) or not all(
        isinstance(tokenization.get(field), bool) for field in ("add_bos", "add_eos")
    ):
        raise ProvenanceValidationError(
            "Data manifest tokenization must define boolean add_bos and add_eos values."
        )
    if not isinstance(split, Mapping) or "strategy" not in split or "train_fraction" not in split:
        raise ProvenanceValidationError(
            "Data manifest preprocessing must include split strategy and train_fraction."
        )
    if not isinstance(split["strategy"], str) or not split["strategy"]:
        raise ProvenanceValidationError("Data manifest split strategy must be non-empty text.")
    train_fraction = split["train_fraction"]
    if (
        isinstance(train_fraction, bool)
        or not isinstance(train_fraction, (int, float))
        or not 0 < float(train_fraction) < 1
    ):
        raise ProvenanceValidationError(
            "Data manifest split train_fraction must be numeric and between 0 and 1."
        )
    stored_digest = _validate_digest(manifest.get("digest"), "data manifest digest")
    expected_digest = canonical_sha256(_data_manifest_digest_payload(manifest))
    if stored_digest != expected_digest:
        raise ProvenanceValidationError(
            "Data manifest digest does not match its canonical identity fields."
        )


def compare_data_manifests(
    checkpoint_manifest: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
) -> None:
    validate_data_manifest(checkpoint_manifest)
    validate_data_manifest(runtime_manifest)
    compare_tokenizer_identities(
        checkpoint_manifest["tokenizer"], runtime_manifest["tokenizer"]
    )
    if checkpoint_manifest["digest"] == runtime_manifest["digest"]:
        return

    checkpoint_artifacts = {
        (item["role"], item["logical_name"]): item
        for item in checkpoint_manifest["artifacts"]
    }
    runtime_artifacts = {
        (item["role"], item["logical_name"]): item
        for item in runtime_manifest["artifacts"]
    }
    for key in sorted(set(checkpoint_artifacts) | set(runtime_artifacts)):
        if checkpoint_artifacts.get(key) != runtime_artifacts.get(key):
            component = key[0] if key in checkpoint_artifacts or key in runtime_artifacts else "data"
            raise ProvenanceMismatchError(
                component,
                f"Data fingerprint mismatch for {component}: "
                f"checkpoint={short_digest(checkpoint_manifest['digest'])}, "
                f"runtime={short_digest(runtime_manifest['digest'])}.",
            )
    if checkpoint_manifest["preprocessing"] != runtime_manifest["preprocessing"]:
        raise ProvenanceMismatchError(
            "split configuration",
            "Data fingerprint mismatch for split configuration: "
            f"checkpoint={short_digest(checkpoint_manifest['digest'])}, "
            f"runtime={short_digest(runtime_manifest['digest'])}.",
        )
    raise ProvenanceMismatchError(
        "data manifest",
        "Data manifest canonical identity differs: "
        f"checkpoint={short_digest(checkpoint_manifest['digest'])}, "
        f"runtime={short_digest(runtime_manifest['digest'])}.",
    )


def build_checkpoint_provenance(
    tokenizer_identity: Mapping[str, Any],
    *,
    data_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_tokenizer_identity(tokenizer_identity)
    provenance: dict[str, Any] = {
        "version": CHECKPOINT_PROVENANCE_VERSION,
        "tokenizer": dict(tokenizer_identity),
    }
    if data_manifest is not None:
        validate_data_manifest(data_manifest)
        compare_tokenizer_identities(tokenizer_identity, data_manifest["tokenizer"])
        provenance["data_manifest"] = dict(data_manifest)
        provenance["data_manifest_digest"] = data_manifest["digest"]
    return provenance


def validate_checkpoint_provenance(
    provenance: Mapping[str, Any],
    *,
    require_data: bool,
) -> None:
    if not isinstance(provenance, Mapping):
        raise ProvenanceValidationError("Checkpoint provenance must be a mapping.")
    _validate_version(
        provenance, "checkpoint provenance", CHECKPOINT_PROVENANCE_VERSION
    )
    validate_tokenizer_identity(provenance.get("tokenizer"))
    data_manifest = provenance.get("data_manifest")
    if require_data and data_manifest is None:
        raise ProvenanceValidationError(
            "Pretraining checkpoint provenance is missing data_manifest."
        )
    if data_manifest is not None:
        validate_data_manifest(data_manifest)
        compare_tokenizer_identities(
            provenance["tokenizer"], data_manifest["tokenizer"]
        )
        if provenance.get("data_manifest_digest") != data_manifest["digest"]:
            raise ProvenanceValidationError(
                "Checkpoint provenance data_manifest_digest does not match data_manifest.digest."
            )
    elif "data_manifest_digest" in provenance:
        raise ProvenanceValidationError(
            "Checkpoint provenance has data_manifest_digest without data_manifest."
        )


def write_data_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_data_manifest(manifest)
    output = Path(path)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def short_digest(value: str, length: int = 12) -> str:
    digest = _validate_digest(value, "digest")
    return digest[:length]


def _tokenizer_digest_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": identity.get("version"),
        "algorithm": identity.get("algorithm"),
        "model": dict(identity.get("model", {})),
        "vocab_size": identity.get("vocab_size"),
        "special_tokens": dict(identity.get("special_tokens", {})),
    }


def _data_manifest_digest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts", [])
    normalized_artifacts = sorted(
        (_normalized_artifact(item) for item in artifacts),
        key=lambda item: (item["role"], item["logical_name"]),
    )
    return {
        "version": manifest.get("version"),
        "algorithm": manifest.get("algorithm"),
        "tokenizer": dict(manifest.get("tokenizer", {})),
        "artifacts": normalized_artifacts,
        "preprocessing": dict(manifest.get("preprocessing", {})),
    }


def _normalized_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise ProvenanceValidationError("Data manifest artifact must be a mapping.")
    role = artifact.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ProvenanceValidationError("Data manifest artifact role must be non-empty text.")
    logical_name = normalize_logical_name(artifact.get("logical_name", ""))
    file_format = artifact.get("format")
    dtype = artifact.get("dtype")
    if not isinstance(file_format, str) or not file_format:
        raise ProvenanceValidationError("Data manifest artifact format must be non-empty text.")
    if not isinstance(dtype, str) or not dtype:
        raise ProvenanceValidationError("Data manifest artifact dtype must be non-empty text.")
    return {
        "role": role.strip(),
        "logical_name": logical_name,
        "format": file_format,
        "size_bytes": _non_negative_int(
            artifact.get("size_bytes"), f"{role} size_bytes"
        ),
        "sha256": _validate_digest(artifact.get("sha256"), f"{role} sha256"),
        "token_count": _non_negative_int(
            artifact.get("token_count"), f"{role} token_count"
        ),
        "dtype": dtype,
    }


def _validated_special_tokens(value: Mapping[str, int] | Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ProvenanceValidationError("Tokenizer special_tokens must be a mapping.")
    missing = [token for token in REQUIRED_SPECIAL_TOKENS if token not in value]
    if missing:
        raise ProvenanceValidationError(
            "Tokenizer identity is missing required special tokens: "
            + ", ".join(missing)
            + "."
        )
    normalized = {
        token: _non_negative_int(value[token], f"special token {token!r}")
        for token in REQUIRED_SPECIAL_TOKENS
    }
    if len(set(normalized.values())) != len(normalized):
        raise ProvenanceValidationError(
            "Tokenizer required special tokens must map to distinct token IDs."
        )
    return normalized


def _validate_version(
    value: Mapping[str, Any],
    label: str,
    supported: int,
) -> None:
    version = value.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ProvenanceValidationError(f"{label} version must be an integer.")
    if version != supported:
        raise ProvenanceValidationError(
            f"Unsupported {label} version {version}; this ByteSeed build supports version {supported}."
        )


def _validate_algorithm(value: Mapping[str, Any], label: str) -> None:
    algorithm = value.get("algorithm")
    if algorithm != HASH_ALGORITHM:
        raise ProvenanceValidationError(
            f"Unsupported {label} hash algorithm {algorithm!r}; expected {HASH_ALGORITHM}."
        )


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProvenanceValidationError(
            f"{label} must be a lowercase 64-character SHA-256 digest."
        )
    return value


def _non_negative_int(value: Any, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ProvenanceValidationError(f"{label} must be a {qualifier} integer.")
    return value


def _file_size(path: Path, label: str) -> int:
    if not path.exists():
        raise FileNotFoundError(f"{label} file is missing: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label} path must be a file: {path}")
    try:
        return path.stat().st_size
    except OSError as exc:
        raise ProvenanceError(f"Could not stat {label} file {path}: {exc}") from exc
