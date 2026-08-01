"""Durable, integrity-checked storage for native-study artifacts.

The checkpoint file is deliberately a small JSONL protocol: each saved case is
an independently hashable envelope, so a restart never trusts an aggregate
summary or a partially-written record.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping
from contextlib import contextmanager


class CheckpointError(ValueError):
    """Base class for invalid checkpoint data."""


class DuplicateRecordError(CheckpointError):
    """Raised when an append would reuse a completed case identifier."""


class CorruptCheckpointError(CheckpointError):
    """Raised when a checkpoint cannot be validated without losing data."""


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize a mapping in a stable, hashable JSON representation."""
    if not isinstance(value, Mapping):
        raise TypeError("canonical_json requires a mapping")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def record_digest(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of a record's canonical JSON payload."""
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def redact_secrets(value: Any) -> Any:
    """Return a deep copy with values under credential-like keys redacted."""
    if isinstance(value, Mapping):
        return {
            key: "[REDACTED]" if _is_secret_key(key) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    return value


def _is_secret_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower().replace("-", "_")
    if normalized in {
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "token",
        "secret",
        "client_secret",
        "password",
        "credential",
        "credentials",
        "private_key",
        "hf_token",
        "huggingface_token",
    }:
        return True
    return normalized.endswith(("_api_key", "_token", "_secret", "_password", "_private_key"))


@dataclass(frozen=True, init=False)
class StudyManifest:
    """A redacted aggregate payload that can be atomically persisted.

    ``payload`` is intentionally generic: stages can add deterministic summary
    metadata without teaching this integrity layer about their domain schema.
    The stored envelope hashes the complete redacted payload.
    """

    _payload_json: str
    sha256: str

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("StudyManifest payload must be a mapping")
        # Keep the canonical JSON, not a caller-visible mutable mapping.  Each
        # public accessor receives a fresh detached copy of this snapshot.
        payload_json = canonical_json(redact_secrets(payload))
        object.__setattr__(self, "_payload_json", payload_json)
        object.__setattr__(self, "sha256", sha256(payload_json.encode("utf-8")).hexdigest())

    @property
    def payload(self) -> dict[str, Any]:
        """Return a detached copy of the manifest payload."""
        return json.loads(self._payload_json)

    @classmethod
    def from_records(
        cls,
        records: Mapping[str, Mapping[str, Any]],
        *,
        expected_case_ids: Iterable[str] = (),
    ) -> "StudyManifest":
        """Build a compact aggregate manifest from validated checkpoint records."""
        record_digests: dict[str, str] = {}
        for case_id, record in records.items():
            if not isinstance(case_id, str) or not isinstance(record, Mapping):
                raise TypeError("records must map string case IDs to mappings")
            record_digests[case_id] = record_digest(record)
        expected = list(expected_case_ids)
        if any(not isinstance(case_id, str) for case_id in expected):
            raise TypeError("expected_case_ids must contain strings")
        return cls(
            {
                "expected_case_ids": expected,
                "completed_case_ids": sorted(record_digests),
                "record_sha256": record_digests,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable integrity envelope."""
        return {"payload": self.payload, "sha256": self.sha256}

    def write(self, path: Path) -> None:
        """Atomically write this manifest through a sibling temporary file."""
        payload = self.payload
        if record_digest(payload) != self.sha256:
            raise ValueError("manifest payload digest mismatch")
        _atomic_write_bytes(Path(path), (canonical_json({"payload": payload, "sha256": self.sha256}) + "\n").encode("utf-8"))

    write_atomic = write


class JsonlCheckpointStore:
    """Append-only, one-record-per-line storage with restart validation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, record: Mapping[str, Any]) -> str:
        """Durably append a uniquely identified record and return its digest."""
        if not isinstance(record, Mapping):
            raise TypeError("checkpoint record must be a mapping")
        payload = redact_secrets(dict(record))
        case_id = payload.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("checkpoint record requires a non-empty string case_id")

        with _checkpoint_lock(self.path):
            existing = self._load_validated()
            if case_id in existing:
                raise DuplicateRecordError(f"duplicate checkpoint case_id: {case_id}")

            digest = record_digest(payload)
            envelope = {"case_id": case_id, "payload": payload, "sha256": digest}
            line = canonical_json(envelope).encode("utf-8") + b"\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return digest

    def load_validated(self) -> dict[str, dict[str, Any]]:
        """Return all records after validating their envelopes and digests.

        A final incomplete or blank physical line is the sole recoverable form
        of damage.  It is copied to ``<checkpoint>.corrupt`` and removed from
        the live file; every other malformed byte sequence remains an error.
        """
        with _checkpoint_lock(self.path):
            return self._load_validated()

    def _load_validated(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise CorruptCheckpointError(f"cannot read checkpoint {self.path}: {exc}") from exc
        if not raw:
            return {}

        lines = raw.splitlines(keepends=True)
        records: dict[str, dict[str, Any]] = {}
        for index, raw_line in enumerate(lines):
            is_final = index == len(lines) - 1
            content = raw_line.rstrip(b"\r\n")
            if is_final and not raw_line.endswith((b"\n", b"\r")):
                self._quarantine_final_line(raw_line, b"".join(lines[:index]))
                return self._load_validated()
            may_quarantine = is_final and not content
            try:
                text = content.decode("utf-8")
                if not text:
                    raise ValueError("empty JSONL line")
                envelope = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                if may_quarantine:
                    self._quarantine_final_line(raw_line, b"".join(lines[:index]))
                    return self._load_validated()
                raise CorruptCheckpointError(
                    f"invalid checkpoint line {index + 1} in {self.path}"
                ) from exc

            case_id, payload = self._validate_envelope(envelope, index + 1)
            if case_id in records:
                raise CorruptCheckpointError(
                    f"duplicate checkpoint case_id {case_id!r} at line {index + 1}"
                )
            records[case_id] = payload
        return records

    def missing(self, expected_ids: Iterable[str]) -> list[str]:
        """Return expected IDs that do not have validated checkpoint records."""
        completed = self.load_validated()
        return [case_id for case_id in expected_ids if case_id not in completed]

    def _validate_envelope(self, envelope: Any, line_number: int) -> tuple[str, dict[str, Any]]:
        if not isinstance(envelope, dict) or set(envelope) != {"case_id", "payload", "sha256"}:
            raise CorruptCheckpointError(f"invalid checkpoint envelope at line {line_number}")
        case_id = envelope["case_id"]
        payload = envelope["payload"]
        supplied_digest = envelope["sha256"]
        if not isinstance(case_id, str) or not case_id:
            raise CorruptCheckpointError(f"invalid case_id at line {line_number}")
        if not isinstance(payload, dict):
            raise CorruptCheckpointError(f"invalid payload at line {line_number}")
        if payload.get("case_id") != case_id:
            raise CorruptCheckpointError(f"case_id mismatch at line {line_number}")
        if not isinstance(supplied_digest, str) or len(supplied_digest) != 64:
            raise CorruptCheckpointError(f"invalid digest at line {line_number}")
        calculated_digest = record_digest(payload)
        if not hmac.compare_digest(supplied_digest, calculated_digest):
            raise CorruptCheckpointError(f"digest mismatch at line {line_number}")
        return case_id, payload

    def _quarantine_final_line(self, fragment: bytes, valid_prefix: bytes) -> None:
        quarantine_path = self.path.with_name(f"{self.path.name}.corrupt")
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        with quarantine_path.open("ab") as handle:
            handle.write(fragment)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_write_bytes(self.path, valid_prefix)


def _atomic_write_bytes(path: Path, contents: bytes) -> None:
    """Durably replace ``path`` using a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


@contextmanager
def _checkpoint_lock(path: Path):
    """Serialize validation, recovery, and append across cooperating workers."""
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
