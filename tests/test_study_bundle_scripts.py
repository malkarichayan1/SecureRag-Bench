"""Tests for ``scripts/export_study_bundle.py`` and ``scripts/import_study_bundle.py``.

Both scripts are exercised as real subprocesses (matching the precedent in
``tests/test_paper_pdf.py``) so these tests prove the actual CLI contract,
not an in-process shortcut around it. The fixture bundle builder is reused
directly from ``tests.test_study_reporting`` rather than duplicated: it is
~200 lines wired to a dozen production modules, and a hand-copied version
would silently drift from the schema Task 1 actually enforces. ``tests`` is
already an importable package elsewhere in this suite (see
``tests/conftest.py``'s ``from tests.helpers import SimulatedToolbox``), so
this is not a new cross-test-module pattern, just the same one applied to a
richer fixture.
"""

from __future__ import annotations

import csv
import json
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pytest

from secure_rag_bench.evaluation.study_reporting import build_paper_tables, validate_study_bundle
from tests.test_study_reporting import corrupt_one_record, fixture_bundle

ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "scripts" / "export_study_bundle.py"
IMPORT_SCRIPT = ROOT / "scripts" / "import_study_bundle.py"


# ---------------------------------------------------------------------------
# CLI-invocation helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScriptResult:
    exit_code: int
    stdout: str
    stderr: str


def run_export(run_root: Path, output: Path) -> ScriptResult:
    completed = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--run-root", str(run_root), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return ScriptResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class ImportResult:
    exit_code: int
    output: Path
    stdout: str
    stderr: str


def run_import(archive: Path, output_dir: Path) -> ImportResult:
    completed = subprocess.run(
        [sys.executable, str(IMPORT_SCRIPT), str(archive), "--output-dir", str(output_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return ImportResult(completed.returncode, output_dir, completed.stdout, completed.stderr)


def export_fixture_bundle(tmp_path: Path, **fixture_kwargs) -> Path:
    """Build a schema-valid fixture bundle and export it to a zip via the CLI."""
    bundle = fixture_bundle(tmp_path, **fixture_kwargs)
    archive = tmp_path / "bundle.zip"
    result = run_export(bundle.root, archive)
    assert result.exit_code == 0, result.stderr
    return archive


def zip_directory_raw(root: Path, archive: Path) -> None:
    """Zip ``root`` without going through the export script's own validation.

    Used to build archives whose *content* is invalid, to prove the import
    script performs its own ``validate_study_bundle`` check rather than
    trusting whatever produced the archive.
    """
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())


def verify_manifest(manifest_path: Path) -> bool:
    """Recompute every digest ``MANIFEST.sha256`` claims and check file-set parity."""
    directory = manifest_path.parent
    listed: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        listed[relative.strip()] = digest
    present = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(listed) != present:
        return False
    return all(sha256((directory / relative).read_bytes()).hexdigest() == digest for relative, digest in listed.items())


# ---------------------------------------------------------------------------
# Step 1: round-trip test (adapted from the plan's illustrative example)
# ---------------------------------------------------------------------------


def test_export_import_round_trip_generates_only_validated_rows(tmp_path: Path) -> None:
    archive = export_fixture_bundle(tmp_path, skipped_optional=True)

    result = run_import(archive, tmp_path / "paper-generated")

    assert result.exit_code == 0, result.stderr
    assert (result.output / "native_validity.tex").exists()
    text = (result.output / "native_validity.tex").read_text(encoding="utf-8")
    # "claude-base-strict_react" is the fixture's skipped optional configuration
    # (model "claude", tier "optional"); it must never reach an empirical table.
    assert "claude-base-strict_react" not in text
    assert "claude" not in text
    assert "qwen-7b-base-strict_react" in text.replace("\\_", "_")
    assert verify_manifest(result.output / "MANIFEST.sha256")


# ---------------------------------------------------------------------------
# Full output-set and content checks
# ---------------------------------------------------------------------------


def test_import_writes_every_declared_output_file(tmp_path: Path) -> None:
    archive = export_fixture_bundle(tmp_path)
    output_dir = tmp_path / "generated"

    result = run_import(archive, output_dir)

    assert result.exit_code == 0, result.stderr
    expected = {
        "native_validity.tex",
        "native_validity.csv",
        "native_validity_execution_asr.csv",
        "adaptive_monitor.tex",
        "adaptive_monitor.csv",
        "ast_compatibility.tex",
        "ast_compatibility.csv",
        "ast_compatibility_rejection_categories.csv",
        "summary.json",
        "MANIFEST.sha256",
    }
    present = {path.name for path in output_dir.iterdir() if path.is_file()}
    assert expected <= present


def test_summary_json_preserves_raw_numeric_values(tmp_path: Path) -> None:
    """summary.json must carry the exact floats/ints build_paper_tables computed."""
    bundle_root = fixture_bundle(tmp_path).root
    expected_tables = build_paper_tables(validate_study_bundle(bundle_root))

    archive = tmp_path / "bundle.zip"
    assert run_export(bundle_root, archive).exit_code == 0
    output_dir = tmp_path / "generated"
    assert run_import(archive, output_dir).exit_code == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["row_counts"]["native_validity"] == len(expected_tables.native_validity)

    expected_row = expected_tables.native_validity[0]
    actual_row = summary["native_validity"][0]
    assert actual_row["protocol_valid_count"] == expected_row.protocol_valid_count
    assert actual_row["protocol_valid_rate"] == expected_row.protocol_valid_rate
    assert actual_row["protocol_valid_wilson_95"]["lower"] == expected_row.protocol_valid_wilson_95["lower"]
    assert actual_row["execution_asr"][0]["defense"] == expected_row.execution_asr[0].defense
    assert actual_row["execution_asr"][0]["rate"] == expected_row.execution_asr[0].rate


def test_native_validity_csv_has_matching_numeric_columns(tmp_path: Path) -> None:
    bundle_root = fixture_bundle(tmp_path).root
    expected_row = build_paper_tables(validate_study_bundle(bundle_root)).native_validity[0]

    archive = tmp_path / "bundle.zip"
    assert run_export(bundle_root, archive).exit_code == 0
    output_dir = tmp_path / "generated"
    assert run_import(archive, output_dir).exit_code == 0

    with (output_dir / "native_validity.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = rows[0]
    assert row["configuration_id"] == expected_row.configuration_id
    assert float(row["protocol_valid_rate"]) == expected_row.protocol_valid_rate
    assert int(row["protocol_valid_count"]) == expected_row.protocol_valid_count


def test_latex_escapes_underscores_and_other_special_characters(tmp_path: Path) -> None:
    archive = export_fixture_bundle(tmp_path)
    output_dir = tmp_path / "generated"
    assert run_import(archive, output_dir).exit_code == 0

    text = (output_dir / "native_validity.tex").read_text(encoding="utf-8")
    # The fixture's configuration_id is "qwen-7b-base-strict_react": the raw
    # (unescaped) underscore must never appear, only the escaped form.
    assert "strict_react" not in text
    assert r"strict\_react" in text
    assert r"\begin{table}" in text
    assert r"\end{table}" in text


# ---------------------------------------------------------------------------
# Export-side validation
# ---------------------------------------------------------------------------


def test_export_rejects_invalid_run_root_and_writes_no_archive(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    corrupt_one_record(bundle)
    archive = tmp_path / "bundle.zip"

    result = run_export(bundle.root, archive)

    assert result.exit_code != 0
    assert not archive.exists()
    assert not (tmp_path / "bundle.zip.tmp").exists()


# ---------------------------------------------------------------------------
# Import-side validation and atomicity
# ---------------------------------------------------------------------------


def test_import_rejects_bundle_that_fails_validation_and_leaves_output_untouched(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    corrupt_one_record(bundle)
    archive = tmp_path / "bundle.zip"
    zip_directory_raw(bundle.root, archive)

    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    result = run_import(archive, output_dir)

    assert result.exit_code != 0
    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert not (output_dir / "native_validity.tex").exists()


def test_import_preserves_existing_output_dir_on_repeat_success(tmp_path: Path) -> None:
    """A second successful import must cleanly replace the first, not merge with it."""
    archive = export_fixture_bundle(tmp_path)
    output_dir = tmp_path / "generated"

    assert run_import(archive, output_dir).exit_code == 0
    stale = output_dir / "leftover_from_a_previous_manual_edit.txt"
    stale.write_text("stale", encoding="utf-8")

    assert run_import(archive, output_dir).exit_code == 0

    assert not stale.exists()
    assert (output_dir / "native_validity.tex").exists()
    assert verify_manifest(output_dir / "MANIFEST.sha256")


@pytest.mark.parametrize(
    "member_name",
    [
        "/etc/passwd",
        "../../evil.txt",
        "bundle.json/../../evil.txt",
        "C:/evil.txt",
    ],
)
def test_import_rejects_path_traversal_and_absolute_members(tmp_path: Path, member_name: str) -> None:
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member_name, "evil payload")

    output_dir = tmp_path / "generated"
    result = run_import(archive, output_dir)

    assert result.exit_code != 0
    assert not output_dir.exists()


def test_import_rejects_symlink_member(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        info = zipfile.ZipInfo("innocuous_name.json")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(info, "/etc/passwd")

    output_dir = tmp_path / "generated"
    result = run_import(archive, output_dir)

    assert result.exit_code != 0
    assert not output_dir.exists()


def test_import_rejects_duplicate_member_names(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("bundle.json", "{}")
        handle.writestr("bundle.json", "{}")

    output_dir = tmp_path / "generated"
    result = run_import(archive, output_dir)

    assert result.exit_code != 0
    assert not output_dir.exists()


def test_import_rejects_directory_entry_member(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("checkpoints/", "")

    output_dir = tmp_path / "generated"
    result = run_import(archive, output_dir)

    assert result.exit_code != 0
    assert not output_dir.exists()


def test_import_rejects_missing_archive(tmp_path: Path) -> None:
    result = run_import(tmp_path / "does-not-exist.zip", tmp_path / "generated")

    assert result.exit_code != 0
    assert not (tmp_path / "generated").exists()


def test_import_rejects_non_zip_file(tmp_path: Path) -> None:
    not_a_zip = tmp_path / "bundle.zip"
    not_a_zip.write_text("not a zip archive", encoding="utf-8")

    result = run_import(not_a_zip, tmp_path / "generated")

    assert result.exit_code != 0
    assert not (tmp_path / "generated").exists()


# ---------------------------------------------------------------------------
# Export archive safety
# ---------------------------------------------------------------------------


def test_export_produces_zip_with_only_relative_posix_arcnames(tmp_path: Path) -> None:
    archive = export_fixture_bundle(tmp_path)

    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert names, "export produced an empty archive"
        for name in names:
            assert not name.startswith("/")
            assert ".." not in Path(name).parts
            assert "\\" not in name
