"""Tests for append-safe native-study artifacts."""

from __future__ import annotations

import json
import subprocess
import sys

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

_CONCURRENT_APPEND_WORKER = """
import sys
from pathlib import Path

from secure_rag_bench.evaluation.study_artifacts import JsonlCheckpointStore


def main() -> int:
    checkpoint_path = Path(sys.argv[1])
    worker_id = sys.argv[2]
    count = int(sys.argv[3])
    store = JsonlCheckpointStore(checkpoint_path)
    for index in range(count):
        store.append({"case_id": f"w{worker_id}-{index}", "worker": worker_id, "index": index})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


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


def test_checkpoint_append_rejects_non_finite_values_as_corrupt(tmp_path) -> None:
    """A record containing NaN/Infinity cannot be canonically hashed; append must
    surface that as a checkpoint-layer error rather than leaking a bare ValueError
    from json.dumps(allow_nan=False)."""
    store = JsonlCheckpointStore(tmp_path / "records.jsonl")

    with pytest.raises(CorruptCheckpointError):
        store.append({"case_id": "a", "score": float("nan")})


def test_checkpoint_load_rejects_non_finite_payload_as_corrupt(tmp_path) -> None:
    """A tampered or foreign-written line whose payload contains NaN/Infinity must
    fail validation as corrupt, not escape as a bare ValueError from record_digest."""
    path = tmp_path / "records.jsonl"
    store = JsonlCheckpointStore(path)
    store.append({"case_id": "a", "score": 1})

    wrapper = json.loads(path.read_text(encoding="utf-8"))
    wrapper["payload"]["score"] = float("nan")
    path.write_text(json.dumps(wrapper) + "\n", encoding="utf-8")

    with pytest.raises(CorruptCheckpointError):
        store.load_validated()


def test_checkpoint_append_survives_concurrent_os_processes(tmp_path) -> None:
    """A broken or absent process-level lock lets concurrent OS-process writers
    interleave partial writes, corrupting the file or losing/duplicating records.
    This spawns real subprocesses (not threads, which would share one process's
    file-descriptor table and never exercise the fcntl/msvcrt cross-process lock)
    that all append to the same checkpoint path at once.
    """
    checkpoint_path = tmp_path / "records.jsonl"
    script_path = tmp_path / "concurrent_append_worker.py"
    script_path.write_text(_CONCURRENT_APPEND_WORKER, encoding="utf-8")

    worker_count = 6
    records_per_worker = 15
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(script_path),
                str(checkpoint_path),
                str(worker_id),
                str(records_per_worker),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for worker_id in range(worker_count)
    ]
    outcomes = [(process, *process.communicate(timeout=60)) for process in processes]

    for process, stdout, stderr in outcomes:
        assert process.returncode == 0, f"worker failed: stdout={stdout!r} stderr={stderr!r}"

    assert not checkpoint_path.with_name("records.jsonl.corrupt").exists()

    lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == worker_count * records_per_worker

    store = JsonlCheckpointStore(checkpoint_path)
    records = store.load_validated()
    expected_case_ids = {
        f"w{worker_id}-{index}"
        for worker_id in range(worker_count)
        for index in range(records_per_worker)
    }
    assert set(records) == expected_case_ids
    assert len(records) == len(expected_case_ids)
    for worker_id in range(worker_count):
        for index in range(records_per_worker):
            record = records[f"w{worker_id}-{index}"]
            assert record["worker"] == str(worker_id)
            assert record["index"] == index
