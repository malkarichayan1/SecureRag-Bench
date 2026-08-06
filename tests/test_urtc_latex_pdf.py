"""Integration coverage for the URTC manuscript build interface."""

from pathlib import Path
import shutil
import subprocess
import sys

from reportlab.pdfgen import canvas

from scripts.verify_urtc_pdf import verify_pdf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIRECTORY = REPOSITORY_ROOT / "paper" / "urtc"
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
