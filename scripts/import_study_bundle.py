"""Import a study bundle archive and generate validated paper-ready inputs.

Usage::

    python scripts/import_study_bundle.py BUNDLE.zip --output-dir paper/generated

Given an archive produced by ``scripts/export_study_bundle.py`` (or any zip
with the same directory schema -- see ``secure_rag_bench.evaluation
.study_reporting``'s module docstring), this script:

1. **Validates every archive member's name before extracting anything.**
   Absolute paths, ``..`` traversal, backslashes, drive letters, symlink
   members, directory entries, and duplicate member names are all rejected
   up front; nothing is written to disk until every name in the archive has
   passed this check.
2. Extracts into a fresh temporary directory.
3. Calls ``validate_study_bundle`` on that directory. This is the *only*
   authority on whether the bundle's content is trustworthy -- schema
   shape, hash chains, gate arithmetic, secret redaction, all of it. A
   validation failure aborts the import with a non-zero exit code and
   leaves ``--output-dir`` completely untouched.
4. Calls ``build_paper_tables`` on the validated bundle and renders
   ``native_validity``/``adaptive_monitor``/``ast_compatibility`` as escaped
   LaTeX tables plus machine-readable CSV, a ``summary.json`` carrying the
   same rows as plain (unrounded) numbers, and a fresh ``MANIFEST.sha256``
   over exactly the files this script wrote (a new manifest for the
   generated directory -- not a copy of the source bundle's manifest).
5. Writes all of that to a staging directory next to ``--output-dir`` and
   replaces ``--output-dir`` with a single ``os.replace`` once every output
   file exists, so a crash or interruption partway through generation can
   never leave ``--output-dir`` half-written.

## Output file shapes (for Task 5 / anything downstream that reads these)

``native_validity.tex`` and ``ast_compatibility.tex`` each contain **two**
``table`` environments back to back: the primary metric table, then a
companion table for that row type's nested detail (execution ASR by
defense; rejection categories by family). ``adaptive_monitor.tex`` contains
one ``table`` environment. Every ``.tex`` file is a complete,
self-contained ``\\begin{table}...\\end{table}`` block (not a bare
``tabular``), built with plain ``\\hline`` rules rather than ``booktabs``,
so it renders standalone without any assumption about the including
document's preamble. Each ``.csv`` file has one row per dataclass row
(``NativeValidityRow``/``AdaptiveMonitorRow``/``ASTCompatibilityRow``), with
Wilson interval bounds flattened into ``..._ci_lower``/``..._ci_upper``
columns; ``NativeValidityRow.execution_asr`` and
``ASTCompatibilityRow.rejection_categories`` are each nested fields, so they
get their own CSV (``native_validity_execution_asr.csv``,
``ast_compatibility_rejection_categories.csv``) rather than being force-fit
into the parent row.

LaTeX text is escaped through exactly one function, ``escape_latex``, for
exactly the seven characters the implementation plan enumerates: ``\\ % _ &
# { }``. It deliberately does not also escape ``$``, ``^``, or ``~`` --
match the plan's literal list, not the full set of LaTeX-special
characters.

## Headline macros and plot data (for paper/urtc's manuscript, Task 5)

Two further output families exist purely so ``paper/urtc/main.tex`` never
hand-writes a validated number:

- ``macros.tex`` -- one ``\\newcommand`` per headline claim
  (``\\NativeBestValidity``, ``\\NativeModelCount``, ``\\AdaptiveFullASR``,
  ``\\ASTBenignUtility``), each computed from the same ``PaperTables`` the
  detail tables render. Unlike the detail tables, which render an explicit
  ``(no rows)`` placeholder for an empty table (a legitimate state -- e.g. no
  configuration cleared the validity gate), a headline macro has no sensible
  empty rendering: "the best validity was (no rows)%" is not a claim anyone
  can cite. Each macro's builder therefore raises ``BundleImportError`` if its
  source table has zero rows, aborting the whole import (leaving
  ``--output-dir`` untouched, per the existing atomic-replace contract) rather
  than emit a placeholder a reader could mistake for a result.
- ``native_validity_plot.tex`` / ``adaptive_results_plot.tex`` -- complete,
  self-contained ``tikzpicture``/``pgfplots`` bodies (not bare
  ``\\addplot`` fragments) for the two new manuscript figures, built the same
  way ``_tex_table`` builds a complete ``table`` environment: this script
  computes the exact bar coordinates and symbolic x-coordinate list, so the
  figure file never has to reconcile a hand-written axis against generated
  data. Empty source tables raise the same way the macros do.

Both are re-derived from ``build_paper_tables``'/``build_plot_series``'
already-validated rows -- neither reads the filesystem or ``bundle.json``
again -- and both are covered by the same ``MANIFEST.sha256`` this script
already writes over every file in the output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

# ``scripts`` has no ``__init__.py`` and is not part of any installed
# distribution. When this file is executed directly -- ``python
# scripts/import_study_bundle.py ...``, exactly how
# ``tests/test_study_bundle_scripts.py`` invokes it as a real subprocess,
# not ``python -m`` -- only ``scripts/``'s own directory lands on
# ``sys.path[0]``, not the repository root; without the root on
# ``sys.path``, ``scripts`` itself is not importable as a *package*, so the
# ``from scripts._manuscript_macros import ...`` below would fail with
# ``ModuleNotFoundError: No module named 'scripts'``. Insert the repository
# root once, defensively (a no-op when already present, e.g. under pytest).
_REPOSITORY_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from secure_rag_bench.evaluation.study_reporting import (
    AdaptiveMonitorRow,
    ASTCompatibilityRow,
    BundleValidationError,
    NativeValidityRow,
    PaperTables,
    ValidatedStudyBundle,
    build_paper_tables,
    build_plot_series,
    validate_study_bundle,
)

# ``BundleImportError``, ``escape_latex``, headline-macro computation, and
# figure-body generation live in ``_manuscript_macros`` (a leaf module with
# no dependency back on this one -- see its docstring) purely to keep this
# script's core extract/validate/write pipeline close to its pre-Task-5
# size. Every name imported here is used below exactly as if it were still
# defined locally; nothing about the public behavior of this script changes.
# The four ``_..._value`` builders are only *called* inside
# ``_manuscript_macros``, but are imported here too so
# ``tests/test_study_bundle_scripts.py``'s existing
# ``from scripts.import_study_bundle import ...`` continues to resolve them
# without needing to know they moved.
from scripts._manuscript_macros import (
    BundleImportError,
    _adaptive_full_asr_value,
    _adaptive_results_plot_tex,
    _ast_benign_utility_value,
    _headline_macro_values,
    _macros_tex,
    _native_best_validity_value,
    _native_model_count_value,
    _native_validity_plot_tex,
    escape_latex,
)


# ---------------------------------------------------------------------------
# Safe archive extraction
#
# ``study_bundle_io``'s path-safety helpers are internal to the validation
# package (see ``study_reporting``'s module docstring: "the two modules
# below it are internal"), so this mirrors the same *pattern* -- POSIX
# relative paths, no absolute paths, no ``..`` -- rather than importing it.
# ---------------------------------------------------------------------------


def _validate_member_name(name: str) -> None:
    if not name or name != name.strip():
        raise BundleImportError(f"{name!r}: invalid archive member name")
    if "\\" in name:
        raise BundleImportError(f"{name!r}: archive member name must be POSIX-style (no backslashes)")
    candidate = PurePosixPath(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BundleImportError(f"{name!r}: archive member escapes the extraction root")
    if len(name) >= 2 and name[1] == ":":
        raise BundleImportError(f"{name!r}: archive member has an absolute path (drive letter)")


def _is_symlink_entry(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Validate every member, then extract. Nothing is written on failure."""
    infos = archive.infolist()
    seen: set[str] = set()
    for info in infos:
        name = info.filename
        if name.endswith("/"):
            raise BundleImportError(f"{name!r}: archive contains a directory entry")
        _validate_member_name(name)
        if _is_symlink_entry(info):
            raise BundleImportError(f"{name!r}: archive contains a symlink member")
        if name in seen:
            raise BundleImportError(f"{name!r}: duplicate archive member")
        seen.add(name)

    destination_resolved = destination.resolve()
    for info in infos:
        target = (destination / info.filename).resolve()
        if target != destination_resolved and destination_resolved not in target.parents:
            raise BundleImportError(f"{info.filename!r}: escapes extraction root after resolution")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, open(target, "wb") as sink:
            shutil.copyfileobj(source, sink)


# ---------------------------------------------------------------------------
# Cell formatting (``escape_latex`` itself is imported from
# ``_manuscript_macros`` above)
# ---------------------------------------------------------------------------


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.2f}"


def _ci_text(wilson: Mapping[str, float] | None) -> str:
    if wilson is None:
        return "n/a"
    return f"[{wilson['lower'] * 100:.1f}, {wilson['upper'] * 100:.1f}]"


def _format_cell(count: int, denominator: int, rate: float | None, wilson: Mapping[str, float] | None) -> str:
    """Render a rate cell as ``NN.NN\\% (count/denominator, 95\\% CI [lo, hi])``.

    Always human-readable display text; the raw ``count``/``denominator``/
    ``rate``/``wilson_95`` values behind it are written separately (never
    reparsed from this string) in the accompanying CSV and ``summary.json``.
    """
    return f"{_pct(rate)}\\% ({count}/{denominator}, 95\\% CI {_ci_text(wilson)})"


def _tex_table(*, caption: str, label: str, column_spec: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        rf"  \caption{{{escape_latex(caption)}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{column_spec}}}",
        r"    \hline",
        "    " + " & ".join(header) + r" \\",
        r"    \hline",
    ]
    for row in rows:
        lines.append("    " + " & ".join(row) + r" \\")
    if not rows:
        lines.append(r"    \multicolumn{" + str(len(header)) + r"}{c}{(no rows)} \\")
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Native validity: .tex + .csv + execution-ASR companion .csv
# ---------------------------------------------------------------------------


def _native_validity_tex(rows: Sequence[NativeValidityRow]) -> str:
    header = [
        "Configuration", "Model", "Tier", "Setting", "Prompt",
        "Held-out Valid", "Gate", "N", "Protocol Valid",
        "Official ASR (valid)", "Official ASR (all)",
    ]
    body = [
        [
            escape_latex(row.configuration_id),
            escape_latex(row.model),
            escape_latex(row.tier),
            escape_latex(row.setting),
            escape_latex(row.prompt_condition),
            _format_cell(
                row.held_out_protocol_valid_count, row.held_out_denominator,
                row.held_out_protocol_valid_rate, row.held_out_protocol_valid_wilson_95,
            ),
            "Pass" if row.gate_passed else "Fail",
            str(row.case_count),
            _format_cell(row.protocol_valid_count, row.case_count, row.protocol_valid_rate, row.protocol_valid_wilson_95),
            _format_cell(
                row.official_asr_valid_count, row.official_asr_valid_denominator,
                row.official_asr_valid_rate, row.official_asr_valid_wilson_95,
            ),
            _format_cell(row.official_asr_all_count, row.case_count, row.official_asr_all_rate, row.official_asr_all_wilson_95),
        ]
        for row in rows
    ]
    main_table = _tex_table(
        caption="Native validity and official attack success rate by configuration.",
        label="tab:native-validity",
        column_spec="lllll" + "c" * 6,
        header=header,
        rows=body,
    )

    execution_header = ["Configuration", "Defense", "Execution ASR"]
    execution_body = [
        [
            escape_latex(row.configuration_id),
            escape_latex(execution.defense),
            _format_cell(execution.count, execution.denominator, execution.rate, execution.wilson_95),
        ]
        for row in rows
        for execution in row.execution_asr
    ]
    execution_table = _tex_table(
        caption="Execution attack success rate by configuration and defense.",
        label="tab:native-execution-asr",
        column_spec="llc",
        header=execution_header,
        rows=execution_body,
    )
    return main_table + "\n" + execution_table


_NATIVE_VALIDITY_CSV_HEADER = [
    "configuration_id", "model", "tier", "model_id", "setting", "prompt_condition",
    "held_out_denominator", "held_out_protocol_valid_count", "held_out_protocol_valid_rate",
    "held_out_protocol_valid_ci_lower", "held_out_protocol_valid_ci_upper", "gate_passed",
    "case_count",
    "syntax_valid_count", "syntax_valid_rate", "syntax_valid_ci_lower", "syntax_valid_ci_upper",
    "protocol_valid_count", "protocol_valid_rate", "protocol_valid_ci_lower", "protocol_valid_ci_upper",
    "official_asr_valid_count", "official_asr_valid_denominator", "official_asr_valid_rate",
    "official_asr_valid_ci_lower", "official_asr_valid_ci_upper",
    "official_asr_all_count", "official_asr_all_rate", "official_asr_all_ci_lower", "official_asr_all_ci_upper",
]


def _bound(wilson: Mapping[str, float] | None, key: str) -> float | None:
    return None if wilson is None else wilson[key]


def _native_validity_csv_rows(rows: Sequence[NativeValidityRow]) -> Iterable[list[Any]]:
    for row in rows:
        yield [
            row.configuration_id, row.model, row.tier, row.model_id, row.setting, row.prompt_condition,
            row.held_out_denominator, row.held_out_protocol_valid_count, row.held_out_protocol_valid_rate,
            _bound(row.held_out_protocol_valid_wilson_95, "lower"), _bound(row.held_out_protocol_valid_wilson_95, "upper"),
            row.gate_passed,
            row.case_count,
            row.syntax_valid_count, row.syntax_valid_rate,
            _bound(row.syntax_valid_wilson_95, "lower"), _bound(row.syntax_valid_wilson_95, "upper"),
            row.protocol_valid_count, row.protocol_valid_rate,
            _bound(row.protocol_valid_wilson_95, "lower"), _bound(row.protocol_valid_wilson_95, "upper"),
            row.official_asr_valid_count, row.official_asr_valid_denominator, row.official_asr_valid_rate,
            _bound(row.official_asr_valid_wilson_95, "lower"), _bound(row.official_asr_valid_wilson_95, "upper"),
            row.official_asr_all_count, row.official_asr_all_rate,
            _bound(row.official_asr_all_wilson_95, "lower"), _bound(row.official_asr_all_wilson_95, "upper"),
        ]


_EXECUTION_ASR_CSV_HEADER = ["configuration_id", "defense", "count", "denominator", "rate", "ci_lower", "ci_upper"]


def _execution_asr_csv_rows(rows: Sequence[NativeValidityRow]) -> Iterable[list[Any]]:
    for row in rows:
        for execution in row.execution_asr:
            yield [
                row.configuration_id, execution.defense, execution.count, execution.denominator, execution.rate,
                _bound(execution.wilson_95, "lower"), _bound(execution.wilson_95, "upper"),
            ]


# ---------------------------------------------------------------------------
# Adaptive monitor: .tex + .csv
# ---------------------------------------------------------------------------


def _adaptive_monitor_tex(rows: Sequence[AdaptiveMonitorRow]) -> str:
    header = [
        "Run", "Family", "Monitor", "N", "Attack", "Control",
        "Retrieval Exposure", "Attempted Target Action", "Halt Rate",
        "Target Effect ASR", "Benign Utility",
    ]
    body = [
        [
            escape_latex(row.run_id), escape_latex(row.family), escape_latex(row.monitor),
            str(row.case_count), str(row.attack_count), str(row.control_count),
            _format_cell(row.retrieval_exposure_count, row.case_count, row.retrieval_exposure_rate, row.retrieval_exposure_wilson_95),
            _format_cell(
                row.attempted_target_action_count, row.case_count,
                row.attempted_target_action_rate, row.attempted_target_action_wilson_95,
            ),
            _format_cell(row.halt_count, row.case_count, row.halt_rate, row.halt_rate_wilson_95),
            _format_cell(
                row.target_effect_asr_count, row.target_effect_asr_denominator,
                row.target_effect_asr, row.target_effect_asr_wilson_95,
            ),
            _format_cell(row.benign_utility_count, row.benign_utility_denominator, row.benign_utility, row.benign_utility_wilson_95),
        ]
        for row in rows
    ]
    return _tex_table(
        caption="Adaptive attack outcomes by family and monitor.",
        label="tab:adaptive-monitor",
        column_spec="lll" + "c" * 8,
        header=header,
        rows=body,
    )


_ADAPTIVE_MONITOR_CSV_HEADER = [
    "run_id", "family", "monitor", "case_count", "attack_count", "control_count",
    "retrieval_exposure_count", "retrieval_exposure_rate", "retrieval_exposure_ci_lower", "retrieval_exposure_ci_upper",
    "attempted_target_action_count", "attempted_target_action_rate",
    "attempted_target_action_ci_lower", "attempted_target_action_ci_upper",
    "halt_count", "halt_rate", "halt_rate_ci_lower", "halt_rate_ci_upper",
    "target_effect_asr_count", "target_effect_asr_denominator", "target_effect_asr",
    "target_effect_asr_ci_lower", "target_effect_asr_ci_upper",
    "benign_utility_count", "benign_utility_denominator", "benign_utility",
    "benign_utility_ci_lower", "benign_utility_ci_upper",
]


def _adaptive_monitor_csv_rows(rows: Sequence[AdaptiveMonitorRow]) -> Iterable[list[Any]]:
    for row in rows:
        yield [
            row.run_id, row.family, row.monitor, row.case_count, row.attack_count, row.control_count,
            row.retrieval_exposure_count, row.retrieval_exposure_rate,
            _bound(row.retrieval_exposure_wilson_95, "lower"), _bound(row.retrieval_exposure_wilson_95, "upper"),
            row.attempted_target_action_count, row.attempted_target_action_rate,
            _bound(row.attempted_target_action_wilson_95, "lower"), _bound(row.attempted_target_action_wilson_95, "upper"),
            row.halt_count, row.halt_rate,
            _bound(row.halt_rate_wilson_95, "lower"), _bound(row.halt_rate_wilson_95, "upper"),
            row.target_effect_asr_count, row.target_effect_asr_denominator, row.target_effect_asr,
            _bound(row.target_effect_asr_wilson_95, "lower"), _bound(row.target_effect_asr_wilson_95, "upper"),
            row.benign_utility_count, row.benign_utility_denominator, row.benign_utility,
            _bound(row.benign_utility_wilson_95, "lower"), _bound(row.benign_utility_wilson_95, "upper"),
        ]


# ---------------------------------------------------------------------------
# AST compatibility: .tex + .csv + rejection-categories companion .csv
# ---------------------------------------------------------------------------


def _ast_compatibility_tex(rows: Sequence[ASTCompatibilityRow]) -> str:
    header = ["Run", "Model", "Family", "N", "Accepted", "Acceptance Rate", "Execution Rate"]
    body = [
        [
            escape_latex(row.run_id), escape_latex(row.model), escape_latex(row.family),
            str(row.case_count), str(row.accepted_count),
            _format_cell(row.accepted_count, row.case_count, row.acceptance_rate, row.acceptance_wilson_95),
            _format_cell(row.execution_count, row.case_count, row.execution_rate, row.execution_wilson_95),
        ]
        for row in rows
    ]
    main_table = _tex_table(
        caption="Restricted-AST compatibility and execution rate by model and family.",
        label="tab:ast-compatibility",
        column_spec="lll" + "c" * 4,
        header=header,
        rows=body,
    )

    rejection_header = ["Run", "Family", "Rejection Category", "Count"]
    rejection_body = [
        [escape_latex(row.run_id), escape_latex(row.family), escape_latex(category), str(count)]
        for row in rows
        for category, count in row.rejection_categories
    ]
    rejection_table = _tex_table(
        caption="Restricted-AST rejection categories by model and family.",
        label="tab:ast-rejection-categories",
        column_spec="lllc",
        header=rejection_header,
        rows=rejection_body,
    )
    return main_table + "\n" + rejection_table


_AST_COMPATIBILITY_CSV_HEADER = [
    "run_id", "model", "family", "case_count", "accepted_count", "acceptance_rate",
    "acceptance_ci_lower", "acceptance_ci_upper", "execution_count", "execution_rate",
    "execution_ci_lower", "execution_ci_upper",
]


def _ast_compatibility_csv_rows(rows: Sequence[ASTCompatibilityRow]) -> Iterable[list[Any]]:
    for row in rows:
        yield [
            row.run_id, row.model, row.family, row.case_count, row.accepted_count, row.acceptance_rate,
            _bound(row.acceptance_wilson_95, "lower"), _bound(row.acceptance_wilson_95, "upper"),
            row.execution_count, row.execution_rate,
            _bound(row.execution_wilson_95, "lower"), _bound(row.execution_wilson_95, "upper"),
        ]


_REJECTION_CATEGORIES_CSV_HEADER = ["run_id", "model", "family", "category", "count"]


def _rejection_categories_csv_rows(rows: Sequence[ASTCompatibilityRow]) -> Iterable[list[Any]]:
    for row in rows:
        for category, count in row.rejection_categories:
            yield [row.run_id, row.model, row.family, category, count]


# ---------------------------------------------------------------------------
# summary.json
# ---------------------------------------------------------------------------


def _build_summary(
    bundle: ValidatedStudyBundle,
    tables: PaperTables,
    *,
    source_archive: str,
    headline_macros: Mapping[str, float | int],
) -> dict[str, Any]:
    """A plain-JSON summary carrying every row's raw (unrounded) numbers.

    ``native_validity``/``adaptive_monitor``/``ast_compatibility`` are exact
    ``dataclasses.asdict`` dumps of the corresponding ``PaperTables`` tuple,
    so nothing here is ever a rounded display string -- a downstream script
    that wants a specific number reads it directly rather than reparsing a
    formatted cell. ``headline_macros`` mirrors ``macros.tex`` as raw floats
    (unrounded, unformatted) keyed by macro name, so a traceability sheet can
    cite ``summary.json["headline_macros"]["NativeBestValidity"]`` as the
    exact source of the LaTeX macro's rounded ``\\%`` string.
    """
    return {
        "schema_version": 1,
        "study_id": bundle.study_id,
        "created_utc": bundle.created_utc,
        "source_archive": source_archive,
        "row_counts": {
            "native_validity": len(tables.native_validity),
            "adaptive_monitor": len(tables.adaptive_monitor),
            "ast_compatibility": len(tables.ast_compatibility),
        },
        "native_validity": [asdict(row) for row in tables.native_validity],
        "adaptive_monitor": [asdict(row) for row in tables.adaptive_monitor],
        "ast_compatibility": [asdict(row) for row in tables.ast_compatibility],
        "headline_macros": dict(headline_macros),
    }


# ---------------------------------------------------------------------------
# Output generation and atomic replace
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


_PLOT_SERIES_HEADERS: Mapping[str, Sequence[str]] = {
    "native_protocol_validity": [
        "configuration_id", "model", "setting", "prompt_condition", "count", "denominator", "rate", "lower", "upper",
    ],
    "adaptive_asr_by_monitor": ["run_id", "family", "monitor", "count", "denominator", "rate", "lower", "upper"],
    "ast_acceptance_by_family": ["run_id", "model", "family", "count", "denominator", "rate", "lower", "upper"],
}


def _write_plot_series_csvs(staging: Path, bundle: ValidatedStudyBundle) -> None:
    """Write ``build_plot_series``'s three flat series as ``plot_<name>.csv``.

    Not consumed by the two new figures directly (those read the richer
    ``native_validity_plot.tex``/``adaptive_results_plot.tex`` this script
    also writes), but kept as plain, independently-checkable CSV so a future
    figure or an external plotting script never has to re-derive a series
    ``build_plot_series`` already exposes.
    """
    series = build_plot_series(bundle)
    for name, header in _PLOT_SERIES_HEADERS.items():
        rows = ([point[key] for key in header] for point in series[name])
        _write_csv(staging / f"plot_{name}.csv", header, rows)


def _write_outputs(staging: Path, bundle: ValidatedStudyBundle, tables: PaperTables, *, source_archive: str) -> None:
    (staging / "native_validity.tex").write_text(_native_validity_tex(tables.native_validity), encoding="utf-8")
    _write_csv(staging / "native_validity.csv", _NATIVE_VALIDITY_CSV_HEADER, _native_validity_csv_rows(tables.native_validity))
    _write_csv(
        staging / "native_validity_execution_asr.csv",
        _EXECUTION_ASR_CSV_HEADER,
        _execution_asr_csv_rows(tables.native_validity),
    )

    (staging / "adaptive_monitor.tex").write_text(_adaptive_monitor_tex(tables.adaptive_monitor), encoding="utf-8")
    _write_csv(
        staging / "adaptive_monitor.csv", _ADAPTIVE_MONITOR_CSV_HEADER, _adaptive_monitor_csv_rows(tables.adaptive_monitor)
    )

    (staging / "ast_compatibility.tex").write_text(_ast_compatibility_tex(tables.ast_compatibility), encoding="utf-8")
    _write_csv(
        staging / "ast_compatibility.csv", _AST_COMPATIBILITY_CSV_HEADER, _ast_compatibility_csv_rows(tables.ast_compatibility)
    )
    _write_csv(
        staging / "ast_compatibility_rejection_categories.csv",
        _REJECTION_CATEGORIES_CSV_HEADER,
        _rejection_categories_csv_rows(tables.ast_compatibility),
    )

    headline_macros = _headline_macro_values(tables)
    (staging / "macros.tex").write_text(_macros_tex(headline_macros), encoding="utf-8")
    (staging / "native_validity_plot.tex").write_text(
        _native_validity_plot_tex(tables.native_validity), encoding="utf-8"
    )
    (staging / "adaptive_results_plot.tex").write_text(
        _adaptive_results_plot_tex(tables.adaptive_monitor), encoding="utf-8"
    )
    _write_plot_series_csvs(staging, bundle)

    summary = _build_summary(bundle, tables, source_archive=source_archive, headline_macros=headline_macros)
    (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _write_manifest(staging: Path) -> None:
    """Write ``MANIFEST.sha256`` over every file this script just generated.

    This is a *new* manifest describing the generated paper-input directory
    -- not a copy of the source bundle's own ``MANIFEST.sha256`` -- so a
    downstream consumer (e.g. the manuscript build) can verify the generated
    directory independently of the source archive.
    """
    lines = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(staging).as_posix()
        digest = sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}")
    (staging / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _replace_directory(staging: Path, destination: Path) -> None:
    """Atomically swap ``destination`` for ``staging``'s contents.

    ``staging`` is created as a sibling of ``destination`` (not under the
    system temp directory) precisely so this rename stays on one filesystem
    and ``os.replace`` is atomic. A pre-existing ``destination`` is moved
    aside first and only removed once the swap has succeeded, so a failure
    partway through restores the original directory instead of leaving
    neither in place.
    """
    backup = destination.parent / f".{destination.name}.backup-{os.getpid()}-{uuid4().hex[:8]}"
    if backup.exists():
        shutil.rmtree(backup)
    moved_original_aside = False
    if destination.exists():
        os.replace(destination, backup)
        moved_original_aside = True
    try:
        os.replace(staging, destination)
    except OSError:
        if moved_original_aside:
            os.replace(backup, destination)
        raise
    if moved_original_aside and backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def import_bundle(archive_path: Path, output_dir: Path) -> Path:
    """Import ``archive_path`` and (re)generate ``output_dir`` from it.

    Raises ``BundleImportError`` for an unsafe archive, ``BundleValidationError``
    (from ``validate_study_bundle``) for a safe-but-invalid bundle, or
    ``zipfile.BadZipFile`` for a file that is not a zip archive at all. In
    every failure case ``output_dir`` is left exactly as it was found.
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    if not archive_path.is_file():
        raise BundleImportError(f"{archive_path}: archive file not found")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="secure_rag_bench_bundle_") as extract_dir_name:
        extract_dir = Path(extract_dir_name)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extract_dir)

        bundle = validate_study_bundle(extract_dir)
        tables = build_paper_tables(bundle)

        staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}-{uuid4().hex[:8]}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            _write_outputs(staging, bundle, tables, source_archive=archive_path.name)
            _write_manifest(staging)
            _replace_directory(staging, output_dir)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to a study bundle .zip")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to (re)generate, e.g. paper/generated")
    args = parser.parse_args(argv)

    try:
        import_bundle(args.archive, args.output_dir)
    except (BundleImportError, BundleValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except zipfile.BadZipFile as exc:
        print(f"error: {args.archive}: not a valid zip archive: {exc}", file=sys.stderr)
        return 1

    print(str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
