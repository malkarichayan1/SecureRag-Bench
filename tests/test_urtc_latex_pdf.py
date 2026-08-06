"""Integration coverage for the URTC manuscript build interface."""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from reportlab.pdfgen import canvas

from scripts.check_generated_manuscript_inputs import (
    REQUIRED_GENERATED_FILES,
    find_missing,
)
from scripts.import_study_bundle import import_bundle
from scripts.verify_urtc_pdf import verify_pdf
from tests.test_study_reporting import fixture_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIRECTORY = REPOSITORY_ROOT / "paper" / "urtc"
GENERATED_DIRECTORY = REPOSITORY_ROOT / "paper" / "generated"
OUTPUT_PDF = REPOSITORY_ROOT / "output" / "pdf" / "securerag_bench_urtc_draft.pdf"


def test_manuscript_source_contains_required_claim_boundaries() -> None:
    """The paper preserves the approved empirical claims and qualifications."""
    source = (MANUSCRIPT_DIRECTORY / "main.tex").read_text(encoding="utf-8")

    assert "15 benign" in source
    assert r"42.86\%" in source
    assert "2,108" in source
    assert "not an official InjecAgent" in source
    assert "preliminary" in source.lower()


def test_manuscript_source_contains_expanded_figures_and_analysis() -> None:
    """The expanded paper includes both figures and the required analysis."""
    source = (MANUSCRIPT_DIRECTORY / "main.tex").read_text(encoding="utf-8")

    assert r"\input{figures/architecture}" in source
    assert r"\input{figures/ablation_results}" in source
    assert r"Table~\ref{tab:attack-protocol}" in source
    assert "retrieval barrier" in source
    assert "invalid outputs" in source


# ---------------------------------------------------------------------------
# Task 5: the manuscript must consume validated (generated) results only.
# ---------------------------------------------------------------------------


def test_manuscript_inputs_generated_validated_tables_and_macros() -> None:
    """``main.tex`` must ``\\input`` the Task 2/5 generated tables, headline
    macros, and figures rather than hand-writing native/adaptive/AST numbers."""
    source = (MANUSCRIPT_DIRECTORY / "main.tex").read_text(encoding="utf-8")

    assert r"\input{../generated/native_validity.tex}" in source
    assert r"\input{../generated/adaptive_monitor.tex}" in source
    assert r"\input{../generated/ast_compatibility.tex}" in source
    assert r"\input{../generated/macros.tex}" in source
    assert r"\input{figures/native_validity}" in source
    assert r"\input{figures/adaptive_results}" in source


def test_manuscript_no_longer_frames_the_pilot_as_the_full_study_state() -> None:
    """The old "50% valid, so the full study never ran" framing must be gone;
    the pilot is now a clearly labeled, compressed preliminary note."""
    source = (MANUSCRIPT_DIRECTORY / "main.tex").read_text(encoding="utf-8")

    assert "50\\% of outputs were valid" not in source
    assert "Preliminary Pilot Note" in source


def test_manuscript_uses_only_headline_macros_for_new_numeric_claims() -> None:
    """New abstract/conclusion claims cite the generated headline macros
    rather than a hand-written number (the plan's Step 3)."""
    source = (MANUSCRIPT_DIRECTORY / "main.tex").read_text(encoding="utf-8")

    for macro in (r"\NativeBestValidity", r"\NativeModelCount", r"\AdaptiveFullASR", r"\ASTBenignUtility"):
        assert macro in source


def test_no_anticipated_result_language() -> None:
    source = (MANUSCRIPT_DIRECTORY / "main.tex").read_text(encoding="utf-8").lower()

    assert "expected to achieve" not in source
    assert "we anticipate" not in source


def test_optional_or_unexecuted_models_are_marked_optional() -> None:
    """Any mention of an unexecuted/optional model must be clearly qualified."""
    source = (MANUSCRIPT_DIRECTORY / "main.tex").read_text(encoding="utf-8")

    assert "Claude" not in source or "optional" in source.lower()


#: Captures the run of ``../`` segments (group 1, so its length is checkable)
#: separately from the target filename (group 2). Matching *any* number of
#: ``../`` deliberately -- rejecting a wrong depth is ``test_every_generated_
#: input_uses_the_correct_relative_depth``'s job below, not this pattern's.
_GENERATED_INPUT_PATTERN = re.compile(r"\\input\{((?:\.\./)+)generated/([^}]+)\}")

#: latexmk compiles from paper/urtc/ (see paper/urtc/Makefile's ``pdf``
#: target: it runs latexmk with cwd already inside paper/urtc/). ``\input``
#: paths inside *any* file that ends up in that compile -- including a file
#: reached transitively through a nested ``\input``, such as
#: figures/native_validity.tex being \input from main.tex -- resolve
#: relative to that one compile-wide cwd, not relative to the directory of
#: the file containing the \input command. paper/generated/ is exactly one
#: level above paper/urtc/, so every \input{.../generated/...} anywhere in
#: the compile must use exactly one '../', regardless of which subdirectory
#: the referencing .tex file physically lives in.
_EXPECTED_GENERATED_INPUT_DEPTH = 1


def _generated_inputs_with_depth() -> list[tuple[Path, str, int]]:
    """``(source_file, target_filename, '../' count)`` for every
    ``\\input{.../generated/X}`` in ``main.tex`` or a ``figures/*.tex`` file."""
    entries: list[tuple[Path, str, int]] = []
    tex_files = [MANUSCRIPT_DIRECTORY / "main.tex", *sorted((MANUSCRIPT_DIRECTORY / "figures").glob("*.tex"))]
    for tex_file in tex_files:
        text = tex_file.read_text(encoding="utf-8")
        for match in _GENERATED_INPUT_PATTERN.finditer(text):
            entries.append((tex_file, match.group(2), match.group(1).count("../")))
    return entries


def _generated_input_targets() -> set[str]:
    """Every filename ``main.tex`` or a ``figures/*.tex`` file ``\\input``s
    from ``paper/generated/``, resolved to a bare filename."""
    return {target for _, target, _ in _generated_inputs_with_depth()}


def test_every_generated_input_uses_the_correct_relative_depth() -> None:
    """A ``figures/*.tex`` file reached via a nested ``\\input`` must still use
    the *compile-relative* one-level ``../generated/...`` path main.tex uses
    directly, not an extra level as if resolving relative to figures/'s own
    location (that extra level was Task 5's original, now-fixed, bug)."""
    entries = _generated_inputs_with_depth()

    assert entries, "expected at least one \\input{.../generated/...} reference"
    for tex_file, target, depth in entries:
        assert depth == _EXPECTED_GENERATED_INPUT_DEPTH, (
            f"{tex_file.relative_to(REPOSITORY_ROOT).as_posix()} has "
            f"\\input{{{'../' * depth}generated/{target}}} using {depth} level(s) of '../'; "
            f"latexmk compiles from paper/urtc/, exactly {_EXPECTED_GENERATED_INPUT_DEPTH} "
            "level above paper/generated/, so every such \\input -- in main.tex or any file it "
            "transitively \\inputs -- must use exactly that many, regardless of which "
            "subdirectory the referencing file lives in"
        )


def test_every_generated_input_target_is_covered_by_the_makefile_guard() -> None:
    """Every file the manuscript ``\\input``s from ``paper/generated/`` must be
    one ``scripts/check_generated_manuscript_inputs.py`` actually checks for,
    or a missing file would fail deep inside ``latexmk`` instead of the guard."""
    targets = _generated_input_targets()

    assert targets, "expected at least one \\input{.../generated/...} reference"
    assert targets <= set(REQUIRED_GENERATED_FILES)


def test_check_generated_manuscript_inputs_reports_every_missing_file(tmp_path: Path) -> None:
    """An unimported ``paper/generated/``-shaped directory (only a
    ``.gitkeep``, exactly this repository's current ``paper/generated/``
    until a real Kaggle bundle is imported) is reported missing every
    required manuscript input."""
    empty_generated = tmp_path / "generated"
    empty_generated.mkdir()
    (empty_generated / ".gitkeep").write_text("", encoding="utf-8")

    missing = find_missing(empty_generated)

    assert set(missing) == set(REQUIRED_GENERATED_FILES)


def _zip_bundle_root(root: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())


def test_fixture_bundle_import_resolves_every_manuscript_input(tmp_path: Path) -> None:
    """A validated *fixture* bundle -- confined entirely to ``tmp_path``, never
    touching ``paper/generated/`` -- produces every file ``main.tex`` and its
    figures ``\\input``. This exercises the real ``\\input`` machinery against
    a real (if synthetic) validated bundle without presenting fixture values
    as manuscript results."""
    bundle = fixture_bundle(tmp_path, skipped_optional=True)
    archive = tmp_path / "bundle.zip"
    _zip_bundle_root(bundle.root, archive)

    output_dir = tmp_path / "generated"
    import_bundle(archive, output_dir)

    assert output_dir != GENERATED_DIRECTORY
    assert find_missing(output_dir) == []
    for target in _generated_input_targets():
        assert (output_dir / target).is_file(), f"import did not produce {target}"


def test_make_pdf_produces_nonempty_urtc_draft() -> None:
    """The public Make target produces the provisional URTC PDF artifact."""
    assert shutil.which("make"), "The test environment must provide GNU Make."

    subprocess.run(
        ["make", "-C", str(MANUSCRIPT_DIRECTORY), "pdf"],
        check=True,
        cwd=REPOSITORY_ROOT,
    )

    assert OUTPUT_PDF.is_file()
    assert OUTPUT_PDF.stat().st_size > 0


def _write_pdf(path: Path, pages: list[str]) -> None:
    document = canvas.Canvas(str(path))
    for text in pages:
        document.drawString(72, 720, text)
        document.showPage()
    document.save()


def test_verify_pdf_accepts_required_qualified_claims(tmp_path: Path) -> None:
    """The verifier accepts a five-page draft with all required boundaries."""
    pdf_path = tmp_path / "valid.pdf"
    _write_pdf(
        pdf_path,
        [
            "2,108 records are included in this preliminary draft.",
            "This is not an official InjecAgent score.",
        ],
    )

    assert verify_pdf(pdf_path) == 2


def test_verify_pdf_rejects_an_unqualified_injecagent_score(tmp_path: Path) -> None:
    """An official-score assertion must retain its negating qualification."""
    pdf_path = tmp_path / "unqualified-score.pdf"
    _write_pdf(
        pdf_path,
        [
            "2,108 records are included in this preliminary draft.",
            "This is not an official InjecAgent evaluation.",
            "This is an official InjecAgent score.",
        ],
    )

    try:
        verify_pdf(pdf_path)
    except ValueError as error:
        assert "unqualified official InjecAgent score" in str(error)
    else:
        raise AssertionError("Expected an unqualified-score verification failure.")


def test_verify_pdf_rejects_more_than_five_pages(tmp_path: Path) -> None:
    """URTC submissions must not exceed the five-page format limit."""
    pdf_path = tmp_path / "six-pages.pdf"
    _write_pdf(
        pdf_path,
        ["2,108 preliminary not an official InjecAgent score."] * 6,
    )

    try:
        verify_pdf(pdf_path)
    except ValueError as error:
        assert "more than five pages" in str(error)
    else:
        raise AssertionError("Expected a page-limit verification failure.")


def test_verify_pdf_rejects_missing_required_boundary(tmp_path: Path) -> None:
    """The qualifying claim text must remain present in the PDF."""
    pdf_path = tmp_path / "missing-boundary.pdf"
    _write_pdf(
        pdf_path,
        ["2,108 records are included in this preliminary draft."],
    )

    try:
        verify_pdf(pdf_path)
    except ValueError as error:
        assert "not an official InjecAgent" in str(error)
    else:
        raise AssertionError("Expected a missing-boundary verification failure.")


def test_urtc_pdf_passes_verification() -> None:
    """The public PDF build passes the standalone verification command."""
    assert shutil.which("make"), "The test environment must provide GNU Make."

    subprocess.run(
        ["make", "-C", str(MANUSCRIPT_DIRECTORY), "pdf"],
        check=True,
        cwd=REPOSITORY_ROOT,
    )
    result = subprocess.run(
        [sys.executable, "scripts/verify_urtc_pdf.py"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pages=" in result.stdout
