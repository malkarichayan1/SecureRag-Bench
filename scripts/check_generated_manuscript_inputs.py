"""Fail loudly if ``paper/generated/`` lacks a file the manuscript ``\\input``s.

``paper/urtc/Makefile``'s ``pdf`` target runs this script before invoking
``latexmk``. Without this guard, an unimported ``paper/generated/`` (which,
absent a real validated Kaggle study bundle, contains nothing but
``.gitkeep``) would either fail deep inside ``pdflatex`` with an opaque
"File `../generated/native_validity.tex' not found" error, or -- worse --
tempt someone into hand-writing placeholder numbers into ``main.tex`` just to
get a PDF out. Both are exactly what the plan's "the build must stop with a
clear message rather than insert substitute result text" requirement rules
out.

This check is a standalone script rather than inline ``Makefile`` shell so it
can be exercised directly from Python tests without requiring ``make`` itself
to be installed (this development environment currently has neither
``make`` nor a LaTeX toolchain on ``PATH``).

Usage::

    python scripts/check_generated_manuscript_inputs.py [--generated-dir DIR]

Exits 0 if every required file is present, 1 (with a message on stderr
naming exactly what is missing and how to fix it) otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: Every file ``paper/urtc/main.tex`` and its figures ``\input`` from
#: ``paper/generated/``. Keep in lockstep with ``main.tex``'s ``\input``
#: statements and ``scripts/import_study_bundle.py``'s ``_write_outputs``.
REQUIRED_GENERATED_FILES = (
    "native_validity.tex",
    "adaptive_monitor.tex",
    "ast_compatibility.tex",
    "macros.tex",
    "native_validity_plot.tex",
    "adaptive_results_plot.tex",
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_DIR = REPOSITORY_ROOT / "paper" / "generated"

_REMEDIATION = (
    "Run 'python scripts/import_study_bundle.py <bundle.zip> "
    "--output-dir paper/generated' with a real, validated Kaggle study "
    "bundle archive before building the manuscript."
)


def find_missing(generated_dir: Path) -> list[str]:
    """Return the subset of ``REQUIRED_GENERATED_FILES`` absent from ``generated_dir``."""
    return [name for name in REQUIRED_GENERATED_FILES if not (generated_dir / name).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Directory to check (default: paper/generated relative to the repository root).",
    )
    args = parser.parse_args(argv)

    missing = find_missing(args.generated_dir)
    if missing:
        print(
            f"error: {args.generated_dir} is missing required manuscript input(s): "
            + ", ".join(missing)
            + ". " + _REMEDIATION,
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
