"""Headline manuscript macros and TikZ/pgfplots figure-body generation.

Split out of ``scripts/import_study_bundle.py`` purely to keep that script's
core extract/validate/write pipeline close to its pre-Task-5 size; the three
concerns below (headline-macro computation, ``macros.tex`` rendering, and
figure-body generation) are otherwise independent of archive handling and
bundle validation.

This module is a **leaf**: it imports only from
``secure_rag_bench.evaluation.study_reporting`` and the standard library,
never from ``scripts.import_study_bundle``. That is also why
``BundleImportError`` and ``escape_latex`` (otherwise general-purpose,
archive-handling-adjacent utilities) live here rather than in
``import_study_bundle.py``: both this module's macro/figure builders and
``import_study_bundle.py``'s archive-safety and detail-table code need them,
and a two-way import between the two script modules would be a real Python
circular import, not just an architectural smell. ``import_study_bundle.py``
imports every public name it needs from here; nothing in the reverse
direction exists.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from secure_rag_bench.evaluation.study_reporting import (
    AdaptiveMonitorRow,
    ASTCompatibilityRow,
    NativeValidityRow,
    PaperTables,
)


class BundleImportError(ValueError):
    """Raised when an archive is unsafe to extract or cannot be imported."""


# ---------------------------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------------------------

#: Exactly the seven characters the plan's Step 4 enumerates. Deliberately
#: excludes ``$``, ``^``, ``~`` even though those are also LaTeX-special --
#: see ``scripts/import_study_bundle.py``'s module docstring.
_LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "%": r"\%",
    "_": r"\_",
    "&": r"\&",
    "#": r"\#",
    "{": r"\{",
    "}": r"\}",
}


def escape_latex(text: str) -> str:
    """Escape ``\\ % _ & # { }`` for safe inclusion in generated LaTeX."""
    return "".join(_LATEX_ESCAPE_MAP.get(character, character) for character in text)


# ---------------------------------------------------------------------------
# Headline macros (macros.tex)
#
# Each builder raises ``BundleImportError`` on an empty source table instead
# of rendering a placeholder -- see ``import_study_bundle``'s module
# docstring's "Headline macros and plot data" section for why this is
# intentionally stricter than the detail tables it writes.
# ---------------------------------------------------------------------------


def _require_rows(rows: Sequence[Any], *, table: str) -> Sequence[Any]:
    if not rows:
        raise BundleImportError(
            f"cannot compute headline manuscript claims: the {table!r} table has no "
            "completed rows in this bundle"
        )
    return rows


def _native_best_validity_value(rows: Sequence[NativeValidityRow]) -> float:
    rows = _require_rows(rows, table="native_validity")
    return max(row.protocol_valid_rate for row in rows)


def _native_model_count_value(rows: Sequence[NativeValidityRow]) -> int:
    rows = _require_rows(rows, table="native_validity")
    return len({row.model for row in rows})


def _adaptive_full_asr_value(rows: Sequence[AdaptiveMonitorRow]) -> float:
    full_monitor_rows = [
        row for row in rows if row.monitor == "full_monitor" and row.target_effect_asr is not None
    ]
    if not full_monitor_rows:
        raise BundleImportError(
            "cannot compute headline manuscript claims: the 'adaptive_monitor' table has no "
            "full_monitor row with a defined target-effect ASR in this bundle"
        )
    return max(row.target_effect_asr for row in full_monitor_rows)


def _ast_benign_utility_value(rows: Sequence[ASTCompatibilityRow]) -> float:
    rows = _require_rows(rows, table="ast_compatibility")
    return min(row.execution_rate for row in rows)


#: Macro name -> raw-value builder. ``_headline_macro_values`` runs every
#: builder eagerly (rather than lazily formatting one at a time) so an empty
#: source table is reported for *every* affected macro in one import attempt,
#: not discovered one ``BundleImportError`` at a time across repeated runs.
_HEADLINE_MACRO_BUILDERS: Mapping[str, Any] = {
    "NativeBestValidity": lambda tables: _native_best_validity_value(tables.native_validity),
    "NativeModelCount": lambda tables: _native_model_count_value(tables.native_validity),
    "AdaptiveFullASR": lambda tables: _adaptive_full_asr_value(tables.adaptive_monitor),
    "ASTBenignUtility": lambda tables: _ast_benign_utility_value(tables.ast_compatibility),
}


def _headline_macro_values(tables: PaperTables) -> dict[str, float | int]:
    """Raw (unrounded) headline values, keyed by macro name.

    Written into ``summary.json["headline_macros"]`` so a traceability sheet
    can point at one JSON key instead of re-deriving what ``macros.tex``'s
    rounded LaTeX string means.
    """
    errors: list[str] = []
    values: dict[str, float | int] = {}
    for name, builder in _HEADLINE_MACRO_BUILDERS.items():
        try:
            values[name] = builder(tables)
        except BundleImportError as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise BundleImportError("; ".join(errors))
    return values


def _format_macro_value(name: str, value: float | int) -> str:
    if name == "NativeModelCount":
        return str(value)
    return f"{value * 100:.2f}\\%"


def _macros_tex(headline_macros: Mapping[str, float | int]) -> str:
    """Render the headline ``\\newcommand`` macros ``main.tex`` may cite.

    ``main.tex`` must use only these macros for *new* numeric claims (the
    plan's Step 3); it must never hand-write a number this script could
    generate instead. Takes the already-computed raw values (see
    ``_headline_macro_values``) rather than ``PaperTables`` directly, so the
    caller decides once whether an empty source table aborts the import --
    this function only formats.
    """
    lines = [
        f"\\newcommand{{\\{name}}}{{{_format_macro_value(name, value)}}}"
        for name, value in headline_macros.items()
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figure plot data (native_validity_plot.tex, adaptive_results_plot.tex)
# ---------------------------------------------------------------------------

#: Fixed, meaningful display order for the three monitor configurations the
#: adaptive study runs (``MonitorConfiguration`` in ``adaptive_runner.py``).
#: A monitor absent from the bundle's rows is simply skipped, never invented.
_MONITOR_ORDER = ("no_monitor", "policy_only", "full_monitor")
_MONITOR_LABELS = {
    "no_monitor": "No monitor",
    "policy_only": "Policy only",
    "full_monitor": "Full monitor",
}
#: Base hue per monitor; ``fill={hue}!{pct}`` / ``draw={hue}!70!black`` mirrors
#: the ``blue!55`` / ``orange!70`` fill idiom already used in
#: ``figures/ablation_results.tex``.
_MONITOR_HUES = {"no_monitor": "gray", "policy_only": "orange", "full_monitor": "blue"}
_MONITOR_FILL_PERCENT = {"no_monitor": 45, "policy_only": 70, "full_monitor": 55}

_PLOT_AXIS_COMMON = (
    "      width=\\columnwidth,\n"
    "      height=4.25cm,\n"
    "      ybar,\n"
    "      bar width=6pt,\n"
    "      ymin=0,\n"
    "      ymax=105,\n"
    "      y tick label style={font=\\scriptsize},\n"
    "      ylabel style={font=\\scriptsize},\n"
    "      nodes near coords,\n"
    "      every node near coord/.append style={font=\\tiny},\n"
    "      grid=major,\n"
    "      grid style={gray!20}\n"
)


def _tikz_bar_axis(
    *,
    ylabel: str,
    x_coords: str,
    rotate: int,
    enlarge: float,
    body: str,
    extra_options: str = "",
) -> str:
    """Build one complete, self-contained ``tikzpicture``/``axis`` bar chart.

    Shared scaffold for both figure generators below, the same way every
    table generator in ``import_study_bundle.py`` funnels through
    ``_tex_table`` there: the open/close structure, sizing, grid, and
    tick-label styling live in exactly one place instead of being repeated
    almost verbatim per figure. ``body`` supplies the ``\\addplot`` line(s)
    (and, for a multi-series chart, a trailing ``\\legend{...}`` line);
    ``extra_options`` supplies any axis options beyond the shared ones (e.g.
    a ``legend style``).
    """
    return (
        "\\begin{tikzpicture}\n"
        "  \\begin{axis}[\n"
        f"      ylabel={{{ylabel}}},\n"
        f"      symbolic x coords={{{x_coords}}},\n"
        "      xtick=data,\n"
        f"      x tick label style={{font=\\scriptsize, rotate={rotate}, anchor=east}},\n"
        f"      enlarge x limits={enlarge},\n"
        f"{extra_options}"
        f"{_PLOT_AXIS_COMMON}"
        "    ]\n"
        f"{body}\n"
        "  \\end{axis}\n"
        "\\end{tikzpicture}\n"
    )


def _native_validity_plot_tex(rows: Sequence[NativeValidityRow]) -> str:
    rows = _require_rows(rows, table="native_validity")
    labels = ",".join(escape_latex(row.configuration_id) for row in rows)
    coordinates = " ".join(
        f"({escape_latex(row.configuration_id)},{row.protocol_valid_rate * 100:.2f})" for row in rows
    )
    body = f"    \\addplot+[fill=blue!55, draw=blue!70!black] coordinates {{{coordinates}}};"
    return _tikz_bar_axis(
        ylabel="Protocol Valid (\\%)", x_coords=labels, rotate=25, enlarge=0.15, body=body
    )


def _adaptive_results_plot_tex(rows: Sequence[AdaptiveMonitorRow]) -> str:
    rows = _require_rows(rows, table="adaptive_monitor")
    families = sorted({row.family for row in rows})
    by_key = {(row.family, row.monitor): row for row in rows}
    present_monitors = [monitor for monitor in _MONITOR_ORDER if any(key[1] == monitor for key in by_key)]
    family_labels = ",".join(escape_latex(family) for family in families)

    plot_lines = []
    legend_entries = []
    for monitor in present_monitors:
        hue = _MONITOR_HUES[monitor]
        fill = f"{hue}!{_MONITOR_FILL_PERCENT[monitor]}"
        draw = f"{hue}!70!black"
        points = []
        for family in families:
            row = by_key.get((family, monitor))
            if row is None or row.target_effect_asr is None:
                continue
            points.append(f"({escape_latex(family)},{row.target_effect_asr * 100:.2f})")
        plot_lines.append(
            f"      \\addplot+[fill={fill}, draw={draw}] coordinates {{{' '.join(points)}}};"
        )
        legend_entries.append(escape_latex(_MONITOR_LABELS[monitor]))

    legend = ",".join(legend_entries)
    body = "\n".join(plot_lines) + f"\n      \\legend{{{legend}}}"
    extra_options = (
        "      legend style={font=\\scriptsize, at={(0.5,1.02)}, anchor=south,"
        " legend columns=3, draw=none},\n"
    )
    return _tikz_bar_axis(
        ylabel="Target-Effect ASR (\\%)",
        x_coords=family_labels,
        rotate=18,
        enlarge=0.25,
        body=body,
        extra_options=extra_options,
    )
