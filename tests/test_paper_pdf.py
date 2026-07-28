"""Regression checks for the locally generated review PDF."""

from pathlib import Path
import subprocess
import sys

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "output" / "pdf" / "securerag_bench_offline_draft.pdf"


def test_review_pdf_contains_current_controlled_results() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_paper_pdf.py"],
        cwd=ROOT,
        check=True,
    )
    text = "\n".join(page.extract_text() or "" for page in PdfReader(PDF_PATH).pages)

    assert "15" in text
    assert "42.86%" in text
    assert "2,108" in text
    assert "seeds 1 through 10" in text
