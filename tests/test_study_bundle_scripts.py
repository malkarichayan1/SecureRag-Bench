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
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import pytest

from secure_rag_bench.evaluation.study_reporting import (
    AdaptiveMonitorRow,
    ASTCompatibilityRow,
    NativeValidityRow,
    PaperTables,
    build_paper_tables,
    validate_study_bundle,
)
from scripts.import_study_bundle import (
    BundleImportError,
    _adaptive_full_asr_value,
    _ast_benign_utility_value,
    _headline_macro_values,
    _native_best_validity_value,
    _native_model_count_value,
)
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
# Task 5: headline macros and figure plot data (macros.tex,
# native_validity_plot.tex, adaptive_results_plot.tex)
# ---------------------------------------------------------------------------

_WILSON = {"lower": 0.0, "upper": 1.0}


def _native_validity_row(**overrides) -> NativeValidityRow:
    base = NativeValidityRow(
        configuration_id="cfg-1",
        model="model-a",
        tier="primary",
        model_id="model-a-id",
        setting="base",
        prompt_condition="strict_react",
        held_out_denominator=50,
        held_out_protocol_valid_count=46,
        held_out_protocol_valid_rate=0.92,
        held_out_protocol_valid_wilson_95=_WILSON,
        gate_passed=True,
        case_count=70,
        syntax_valid_count=68,
        syntax_valid_rate=0.9714,
        syntax_valid_wilson_95=_WILSON,
        protocol_valid_count=64,
        protocol_valid_rate=0.9143,
        protocol_valid_wilson_95=_WILSON,
        official_asr_valid_count=0,
        official_asr_valid_denominator=64,
        official_asr_valid_rate=0.0,
        official_asr_valid_wilson_95=_WILSON,
        official_asr_all_count=0,
        official_asr_all_rate=0.0,
        official_asr_all_wilson_95=_WILSON,
        execution_asr=(),
    )
    return replace(base, **overrides)


def _adaptive_monitor_row(**overrides) -> AdaptiveMonitorRow:
    base = AdaptiveMonitorRow(
        run_id="adaptive-1",
        family="direct_injection",
        monitor="full_monitor",
        case_count=4,
        attack_count=2,
        control_count=2,
        retrieval_exposure_count=2,
        retrieval_exposure_rate=1.0,
        retrieval_exposure_wilson_95=_WILSON,
        attempted_target_action_count=2,
        attempted_target_action_rate=1.0,
        attempted_target_action_wilson_95=_WILSON,
        halt_count=2,
        halt_rate=1.0,
        halt_rate_wilson_95=_WILSON,
        target_effect_asr_count=0,
        target_effect_asr_denominator=2,
        target_effect_asr=0.0,
        target_effect_asr_wilson_95=_WILSON,
        benign_utility_count=2,
        benign_utility_denominator=2,
        benign_utility=1.0,
        benign_utility_wilson_95=_WILSON,
    )
    return replace(base, **overrides)


def _ast_compatibility_row(**overrides) -> ASTCompatibilityRow:
    base = ASTCompatibilityRow(
        run_id="ast-1",
        model="model-a",
        family="email",
        case_count=3,
        accepted_count=2,
        acceptance_rate=0.6667,
        acceptance_wilson_95=_WILSON,
        execution_count=1,
        execution_rate=0.3333,
        execution_wilson_95=_WILSON,
        rejection_categories=(),
    )
    return replace(base, **overrides)


def test_native_best_validity_value_is_the_max_protocol_valid_rate() -> None:
    rows = (
        _native_validity_row(configuration_id="a", protocol_valid_rate=0.5),
        _native_validity_row(configuration_id="b", protocol_valid_rate=0.9),
    )

    assert _native_best_validity_value(rows) == 0.9


def test_native_best_validity_value_raises_a_clear_error_on_an_empty_table() -> None:
    with pytest.raises(BundleImportError, match="native_validity"):
        _native_best_validity_value(())


def test_native_model_count_value_counts_distinct_models_not_configurations() -> None:
    rows = (
        _native_validity_row(configuration_id="a", model="model-a"),
        _native_validity_row(configuration_id="b", model="model-b"),
        _native_validity_row(configuration_id="c", model="model-a"),
    )

    assert _native_model_count_value(rows) == 2


def test_adaptive_full_asr_value_ignores_non_full_monitor_rows() -> None:
    rows = (
        _adaptive_monitor_row(monitor="no_monitor", target_effect_asr=1.0),
        _adaptive_monitor_row(monitor="full_monitor", target_effect_asr=0.25),
    )

    assert _adaptive_full_asr_value(rows) == 0.25


def test_adaptive_full_asr_value_is_the_worst_full_monitor_family() -> None:
    rows = (
        _adaptive_monitor_row(family="direct_injection", monitor="full_monitor", target_effect_asr=0.1),
        _adaptive_monitor_row(family="split_payload", monitor="full_monitor", target_effect_asr=0.4),
    )

    assert _adaptive_full_asr_value(rows) == 0.4


def test_adaptive_full_asr_value_raises_when_no_full_monitor_row_has_a_defined_asr() -> None:
    rows = (
        _adaptive_monitor_row(monitor="full_monitor", target_effect_asr=None, target_effect_asr_denominator=0),
    )

    with pytest.raises(BundleImportError, match="full_monitor"):
        _adaptive_full_asr_value(rows)


def test_ast_benign_utility_value_is_the_min_execution_rate() -> None:
    rows = (
        _ast_compatibility_row(family="email", execution_rate=0.8),
        _ast_compatibility_row(family="banking", execution_rate=0.3),
    )

    assert _ast_benign_utility_value(rows) == 0.3


def test_ast_benign_utility_value_raises_a_clear_error_on_an_empty_table() -> None:
    with pytest.raises(BundleImportError, match="ast_compatibility"):
        _ast_benign_utility_value(())


def test_headline_macro_values_reports_every_empty_table_in_one_error() -> None:
    empty_tables = PaperTables(native_validity=(), adaptive_monitor=(), ast_compatibility=())

    with pytest.raises(BundleImportError) as excinfo:
        _headline_macro_values(empty_tables)

    message = str(excinfo.value)
    assert "NativeBestValidity" in message
    assert "NativeModelCount" in message
    assert "AdaptiveFullASR" in message
    assert "ASTBenignUtility" in message


def test_import_writes_headline_macros_and_figure_plot_data(tmp_path: Path) -> None:
    archive = export_fixture_bundle(tmp_path)
    output_dir = tmp_path / "generated"

    result = run_import(archive, output_dir)

    assert result.exit_code == 0, result.stderr
    for name in (
        "macros.tex",
        "native_validity_plot.tex",
        "adaptive_results_plot.tex",
        "plot_native_protocol_validity.csv",
        "plot_adaptive_asr_by_monitor.csv",
        "plot_ast_acceptance_by_family.csv",
    ):
        assert (output_dir / name).is_file(), f"missing {name}"

    macros_text = (output_dir / "macros.tex").read_text(encoding="utf-8")
    for macro in (r"\NativeBestValidity", r"\NativeModelCount", r"\AdaptiveFullASR", r"\ASTBenignUtility"):
        assert f"\\newcommand{{{macro}}}" in macros_text

    plot_text = (output_dir / "native_validity_plot.tex").read_text(encoding="utf-8")
    assert r"\begin{tikzpicture}" in plot_text
    assert r"\addplot+" in plot_text

    assert verify_manifest(output_dir / "MANIFEST.sha256")


def test_summary_json_headline_macros_match_the_raw_values_macros_tex_rounds(tmp_path: Path) -> None:
    bundle_root = fixture_bundle(tmp_path).root
    expected = _headline_macro_values(build_paper_tables(validate_study_bundle(bundle_root)))

    archive = tmp_path / "bundle.zip"
    assert run_export(bundle_root, archive).exit_code == 0
    output_dir = tmp_path / "generated"
    assert run_import(archive, output_dir).exit_code == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    headline_macros = summary["headline_macros"]

    assert headline_macros.keys() == expected.keys()
    for name, value in expected.items():
        assert headline_macros[name] == pytest.approx(value)


def test_import_aborts_and_leaves_output_untouched_when_native_validity_table_is_empty(tmp_path: Path) -> None:
    """A configuration whose pilot never clears the gate produces an empty
    ``native_validity`` table; the import must abort with a clear message
    naming the affected macros instead of writing manuscript inputs with a
    missing headline claim (the plan's "stop with a clear message rather
    than insert substitute result text")."""
    archive = export_fixture_bundle(tmp_path, qualifying=False)
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    result = run_import(archive, output_dir)

    assert result.exit_code != 0
    assert "native_validity" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert not (output_dir / "macros.tex").exists()


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
