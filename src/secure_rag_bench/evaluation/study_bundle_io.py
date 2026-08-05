"""Read-only bundle I/O, integrity primitives, and secret scanning.

The lowest layer of study-bundle validation: everything here is generic over
*which* artifact is being read. It knows about file manifests, JSONL envelopes,
SHA-256 digests, credential-shaped keys, and JSON shape assertions -- and
nothing about native runs, AST runs, or adaptive runs.

Layering (strictly one-way, no cycles)::

    study_reporting          public entry point, tables, plot series
      -> study_bundle_validation   artifact-specific validation chains
           -> study_bundle_io      this module

See ``study_reporting``'s module docstring for the normative bundle schema.

Every function here is read-only. Nothing in this module writes, locks,
quarantines, or repairs a file: the bundle under inspection is exactly as it
was when validation started, which is what makes the integrity results
meaningful.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from secure_rag_bench.evaluation.study_artifacts import record_digest, redact_secrets

BUNDLE_INDEX_NAME = "bundle.json"
FILE_MANIFEST_NAME = "MANIFEST.sha256"
REDACTION_SENTINEL = "[REDACTED]"

_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#: Fields holding verbatim benchmark or model text, exempt from the heuristic
#: token-shaped *value* scan. See point 12 of "What validation actually checks"
#: in ``study_reporting``'s module docstring.
_VERBATIM_TEXT_KEYS = frozenset(
    {
        "system_prompt",
        "rendered_prompt",
        "rendered_prompt_step_2",
        "raw_output",
        "raw_output_step_2",
        "output",
        "output Step 2",
        "Tool Response",
        "Attacker Tool Response",
        "Attacker Instruction",
        "Attacker Tool Parameters",
        "Tool Parameters",
        "Thought",
        "Attacker Tool Thought",
        "detail",
        "halt_reason",
        "invalid_reason",
    }
)

_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}"),
)

_SECRET_PROBE = "__secure_rag_bench_secret_probe__"


class BundleValidationError(ValueError):
    """Raised when a study bundle cannot be trusted to produce paper numbers."""


# ---------------------------------------------------------------------------
# File-level integrity
# ---------------------------------------------------------------------------


def validate_file_manifest(root: Path) -> dict[str, str]:
    manifest_path = root / FILE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleValidationError(f"{FILE_MANIFEST_NAME}: missing bundle file manifest")
    digests: dict[str, str] = {}
    for number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or not _DIGEST_PATTERN.match(digest):
            raise BundleValidationError(
                f"{FILE_MANIFEST_NAME}: line {number} is not '<sha256>  <path>'"
            )
        relative = relative.strip()
        _require_safe_relative_path(relative, FILE_MANIFEST_NAME)
        if relative in digests:
            raise BundleValidationError(
                f"{FILE_MANIFEST_NAME}: duplicate entry for {relative}"
            )
        digests[relative] = digest

    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != FILE_MANIFEST_NAME
    }
    for relative in sorted(set(digests) - present):
        raise BundleValidationError(f"{relative}: listed in {FILE_MANIFEST_NAME} but absent")
    for relative in sorted(present - set(digests)):
        raise BundleValidationError(f"{relative}: present but missing from {FILE_MANIFEST_NAME}")
    for relative, digest in sorted(digests.items()):
        actual = sha256((root / relative).read_bytes()).hexdigest()
        if actual != digest:
            raise BundleValidationError(
                f"{relative}: file sha256 mismatch ({actual} != {digest})"
            )
    return digests


def _require_safe_relative_path(relative: str, label: str) -> None:
    if not relative:
        raise BundleValidationError(f"{label}: empty path")
    if relative != relative.strip() or "\\" in relative:
        raise BundleValidationError(f"{label}: {relative!r} must be a POSIX relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise BundleValidationError(f"{label}: {relative!r} escapes the bundle root")


# ---------------------------------------------------------------------------
# Loading primitives
# ---------------------------------------------------------------------------


def load_json(root: Path, relative: str) -> dict[str, Any]:
    _require_safe_relative_path(relative, BUNDLE_INDEX_NAME)
    path = root / relative
    if not path.is_file():
        raise BundleValidationError(f"{relative}: declared file is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"{relative}: unreadable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BundleValidationError(f"{relative}: expected a JSON object")
    assert_no_unredacted_secrets(payload, label=relative)
    return payload


def load_checkpoint_records(root: Path, relative: str) -> list[dict[str, Any]]:
    """Validate one JSONL checkpoint's envelopes without mutating the bundle.

    Mirrors ``JsonlCheckpointStore``'s envelope contract exactly, but never
    locks or rewrites the file -- see the module docstring's closing note.
    """
    _require_safe_relative_path(relative, BUNDLE_INDEX_NAME)
    path = root / relative
    if not path.is_file():
        raise BundleValidationError(f"{relative}: declared checkpoint is missing")
    raw = path.read_bytes()
    if not raw:
        raise BundleValidationError(f"{relative}: checkpoint is empty")
    if not raw.endswith(b"\n"):
        raise BundleValidationError(f"{relative}: checkpoint has an unterminated final line")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line:
            raise BundleValidationError(f"{relative}: blank checkpoint line {number}")
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BundleValidationError(f"{relative}: invalid JSON on line {number}: {exc}") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"case_id", "payload", "sha256"}:
            raise BundleValidationError(f"{relative}: invalid envelope on line {number}")
        case_id = envelope["case_id"]
        payload = envelope["payload"]
        if not isinstance(case_id, str) or not case_id or not isinstance(payload, dict):
            raise BundleValidationError(f"{relative}: invalid envelope fields on line {number}")
        if payload.get("case_id") != case_id:
            raise BundleValidationError(f"{relative}: case_id mismatch on line {number}")
        if case_id in seen:
            raise BundleValidationError(f"{relative}: duplicate case_id {case_id!r} on line {number}")
        seen.add(case_id)
        actual = record_digest(payload)
        if actual != envelope["sha256"]:
            raise BundleValidationError(
                f"{relative}: envelope sha256 mismatch on line {number} "
                f"({actual} != {envelope['sha256']})"
            )
        assert_no_unredacted_secrets(payload, label=f"{relative}:{number}")
        validate_record_digest(payload, label=f"{relative}:{number}")
        records.append(payload)
    return records


def validate_record_digest(record: Mapping[str, Any], *, label: str) -> None:
    claimed = record.get("record_sha256")
    if not isinstance(claimed, str) or not _DIGEST_PATTERN.match(claimed):
        raise BundleValidationError(f"{label}: record is missing a valid record_sha256")
    unhashed = {key: value for key, value in record.items() if key != "record_sha256"}
    actual = record_digest(unhashed)
    if actual != claimed:
        raise BundleValidationError(
            f"{label}: record_sha256 mismatch for case {record.get('case_id')!r} "
            f"({actual} != {claimed})"
        )


# ---------------------------------------------------------------------------
# Secret scanning
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _is_secret_key(key: str) -> bool:
    """Whether ``redact_secrets`` would replace this key's value.

    Probed through ``redact_secrets`` itself rather than duplicating its
    keyword list, so the two can never disagree about what counts as a
    credential-shaped key.
    """
    return redact_secrets({key: _SECRET_PROBE}).get(key) == REDACTION_SENTINEL


def assert_no_unredacted_secrets(payload: Any, *, label: str) -> None:
    """Reject credential-shaped keys and token-shaped values anywhere inside."""
    stack: list[tuple[str, Any, bool]] = [("", payload, False)]
    while stack:
        path, node, verbatim = stack.pop()
        if isinstance(node, Mapping):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                if isinstance(key, str) and _is_secret_key(key) and value != REDACTION_SENTINEL:
                    raise BundleValidationError(
                        f"{label}: credential-shaped key {child!r} is not redacted"
                    )
                stack.append((child, value, isinstance(key, str) and key in _VERBATIM_TEXT_KEYS))
        elif isinstance(node, list):
            for position, item in enumerate(node):
                stack.append((f"{path}[{position}]", item, verbatim))
        elif isinstance(node, str) and not verbatim:
            for pattern in _SECRET_VALUE_PATTERNS:
                if pattern.search(node):
                    raise BundleValidationError(
                        f"{label}: token-shaped secret value at {path or '<root>'!r}"
                    )


# ---------------------------------------------------------------------------
# Index parsing helpers
# ---------------------------------------------------------------------------


def require_list(payload: Mapping[str, Any], key: str, label: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise BundleValidationError(f"{label}: {key!r} must be a list")
    return value


def require_text(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BundleValidationError(f"{label}: {key!r} must be a non-empty string")
    return value


def require_int(payload: Mapping[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BundleValidationError(f"{label}: {key!r} must be an integer")
    return value


def require_timestamp(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = require_text(payload, key, label)
    if not _TIMESTAMP_PATTERN.match(value):
        raise BundleValidationError(
            f"{label}: {key!r} must be an ISO-8601 UTC timestamp ending in 'Z'"
        )
    return value


def require_unique(values: Iterable[str], *, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise BundleValidationError(f"{BUNDLE_INDEX_NAME}: duplicate {label} {value!r}")
        seen.add(value)


def string_list(payload: Mapping[str, Any], key: str, label: str) -> list[str]:
    values = require_list(payload, key, label)
    if any(not isinstance(value, str) or not value for value in values):
        raise BundleValidationError(f"{label}: {key!r} must contain non-empty strings")
    return list(values)


def require_mapping(payload: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BundleValidationError(f"{label}: {key!r} must be an object")
    return value
