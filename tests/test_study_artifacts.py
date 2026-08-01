"""Tests for append-safe native-study artifacts."""

from __future__ import annotations

import json

import pytest

from secure_rag_bench.evaluation.study_artifacts import (
    CorruptCheckpointError,
    DuplicateRecordError,
    JsonlCheckpointStore,
    StudyManifest,
    canonical_json,
    record_digest,
    redact_secrets,
)


def test_canonical_json_and_digest_ignore_mapping_order() -> None:
    """A change to canonical ordering or encoding makes this determinism check fail."""
    first = {"case_id": "a", "metadata": {"model": "demo", "temperature": 0}}
    second = {"metadata": {"temperature": 0, "model": "demo"}, "case_id": "a"}

    assert canonical_json(first) == '{"case_id":"a","metadata":{"model":"demo","temperature":0}}'
    assert record_digest(first) == record_digest(second)


def test_checkpoint_round_trip_detects_duplicates_and_corruption(tmp_path) -> None:
    """A missing duplicate or per-record validation branch makes this test fail."""
    store = JsonlCheckpointStore(tmp_path / "records.jsonl")
    store.append({"case_id": "a", "raw_output": "x"})

    assert store.load_validated()["a"]["raw_output"] == "x"
    assert store.missing(["b", "a", "c"]) == ["b", "c"]
    with pytest.raises(DuplicateRecordError):
        store.append({"case_id": "a", "raw_output": "y"})

    (tmp_path / "records.jsonl").write_text(
        '{"case_id":"a"}\nBROKEN\n', encoding="utf-8"
    )
    with pytest.raises(CorruptCheckpointError):
        store.load_validated()


def test_checkpoint_quarantines_only_a_truncated_final_line(tmp_path) -> None:
    """Treating every corrupt line as recoverable would make the interior assertion fail."""
    path = tmp_path / "records.jsonl"
    store = JsonlCheckpointStore(path)
    store.append({"case_id": "a", "raw_output": "x"})
    path.write_bytes(path.read_bytes() + b'{"case_id":"unfinished"')

    assert store.load_validated()["a"]["raw_output"] == "x"
    assert path.with_name("records.jsonl.corrupt").read_bytes() == b'{"case_id":"unfinished"'
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    path.write_bytes(path.read_bytes() + b"BROKEN\n")
    with pytest.raises(CorruptCheckpointError):
        store.load_validated()
    assert path.with_name("records.jsonl.corrupt").read_bytes() == b'{"case_id":"unfinished"'


def test_checkpoint_quarantines_even_a_valid_unterminated_final_record(tmp_path) -> None:
    """Accepting a final record without append's newline can mark an interrupted write complete."""
    path = tmp_path / "records.jsonl"
    store = JsonlCheckpointStore(path)
    store.append({"case_id": "a", "raw_output": "x"})
    unterminated_line = path.read_bytes().rstrip(b"\n")
    path.write_bytes(unterminated_line)

    assert store.load_validated() == {}
    assert path.with_name("records.jsonl.corrupt").read_bytes() == unterminated_line
    assert path.read_bytes() == b""


def test_checkpoint_rejects_tampered_digest_and_mismatched_case_id(tmp_path) -> None:
    """Skipping hash or envelope/payload identity checks makes this test fail."""
    path = tmp_path / "records.jsonl"
    store = JsonlCheckpointStore(path)
    store.append({"case_id": "a", "raw_output": "x"})
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    wrapper["sha256"] = "0" * 64
    path.write_text(json.dumps(wrapper) + "\n", encoding="utf-8")

    with pytest.raises(CorruptCheckpointError):
        store.load_validated()

    wrapper["sha256"] = record_digest(wrapper["payload"])
    wrapper["payload"]["case_id"] = "other"
    path.write_text(json.dumps(wrapper) + "\n", encoding="utf-8")
    with pytest.raises(CorruptCheckpointError):
        store.load_validated()


def test_secret_redaction_is_recursive() -> None:
    """Removing recursive traversal or secret-key matching makes this test fail."""
    assert redact_secrets(
        {
            "Authorization": "Bearer abc",
            "nested": {"api_key": "abc", "values": [{"TOKEN": "xyz"}]},
        }
    ) == {
        "Authorization": "[REDACTED]",
        "nested": {"api_key": "[REDACTED]", "values": [{"TOKEN": "[REDACTED]"}]},
    }


def test_checkpoint_redacts_credential_fields_before_persisting(tmp_path) -> None:
    """Writing the supplied record verbatim would leak checkpoint credentials."""
    path = tmp_path / "records.jsonl"
    store = JsonlCheckpointStore(path)
    store.append({"case_id": "a", "Authorization": "Bearer abc", "nested": {"api_key": "abc"}})

    payload = store.load_validated()["a"]
    assert payload["Authorization"] == "[REDACTED]"
    assert payload["nested"]["api_key"] == "[REDACTED]"
    assert "Bearer abc" not in path.read_text(encoding="utf-8")


def test_manifest_is_redacted_hashed_and_atomically_serialized(tmp_path) -> None:
    """A non-atomic writer or an unredacted manifest changes the persisted result."""
    path = tmp_path / "summary.json"
    manifest = StudyManifest({"api_key": "secret", "records": {"a": {"score": 1}}})
    manifest.write(path)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["payload"]["api_key"] == "[REDACTED]"
    assert persisted["sha256"] == record_digest(persisted["payload"])
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_nested_payload_cannot_be_mutated_after_construction(tmp_path) -> None:
    """Exposing the stored nested payload would persist a changed value with an old digest."""
    source = {"records": {"a": {"score": 1}}}
    manifest = StudyManifest(source)
    source["records"]["a"]["score"] = 99
    manifest.payload["records"]["a"]["score"] = 98
    exported = manifest.to_dict()
    exported["payload"]["records"]["a"]["score"] = 97

    path = tmp_path / "summary.json"
    manifest.write(path)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["payload"]["records"]["a"]["score"] == 1
    assert persisted["sha256"] == record_digest(persisted["payload"])
