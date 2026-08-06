"""Verify the URTC submission PDF's format and claim boundaries."""

from __future__ import annotations

from pathlib import Path
import sys

from pypdf import PdfReader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = REPOSITORY_ROOT / "output" / "pdf" / "securerag_bench_urtc_draft.pdf"
REQUIRED_PHRASES = ("2,108", "not an official InjecAgent")
UNQUALIFIED_SCORE = "official InjecAgent score"
QUALIFIED_SCORE = "not an official InjecAgent score"


def verify_pdf(pdf_path: Path = PDF_PATH) -> int:
    """Return the page count when *pdf_path* satisfies URTC constraints.

    Raises:
        ValueError: If the page-count or claim-boundary checks fail.
    """
    reader = PdfReader(pdf_path)
    page_count = len(reader.pages)
    if page_count > 5:
        raise ValueError(f"PDF has more than five pages: pages={page_count}")

    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for phrase in REQUIRED_PHRASES:
        if phrase not in text:
            raise ValueError(f"PDF is missing required text: {phrase}")
    if "preliminary" not in text.lower():
        raise ValueError("PDF is missing required text: preliminary")

    unqualified_scores = text.count(UNQUALIFIED_SCORE) - text.count(QUALIFIED_SCORE)
    if unqualified_scores > 0:
        raise ValueError("PDF contains an unqualified official InjecAgent score")

    return page_count


def main() -> int:
    """Run verification and print the result for build automation."""
    try:
        page_count = verify_pdf()
    except (FileNotFoundError, ValueError) as error:
        print(f"verification=failed: {error}", file=sys.stderr)
        return 1

    print(f"pages={page_count} verification=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
