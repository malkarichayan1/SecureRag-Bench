# URTC LaTeX Manuscript Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a five-page LaTeX manuscript for the SecureRAG-Bench empirical study and a visually verified PDF.

**Architecture:** Add a self-contained LaTeX project under `paper/urtc/`, backed by a dedicated BibTeX file and stable Makefile. Keep the existing ReportLab generator separate as an offline-review artifact. Generate the submission draft under `output/pdf/`, then validate it automatically and visually.

**Tech Stack:** LaTeX (`latexmk`/PDFLaTeX and BibTeX), Python 3.10+, `pytest`, `pypdf`, Poppler.

## Global Constraints

- Work to the supplied URTC 2025 format only provisionally: English, five pages maximum, single-spaced, minimum 10-point font, abstract under 500 words, numbered tables/figures, numbered IEEE references.
- Do not use the 2025 deadline or portal as 2026 submission information.
- Preserve existing user changes outside named files.
- Present an implementation/evaluation inspired by CaMeL, never a claim to invent CaMeL.
- Quantitative claims must be traceable to `docs/claim_traceability.md` and saved artifacts.
- The 2,108-case result is an offline payload-transfer study, never an official InjecAgent score.
- Mention the native Qwen 7B pilot only as preliminary: its 50% validity was below the 90% full-evaluation gate.
- Render and visually inspect the final PDF; it must not exceed five pages.

---

## File Structure

- `paper/urtc/main.tex`: complete manuscript, table, and references.
- `paper/urtc/references.bib`: verified BibTeX references.
- `paper/urtc/Makefile`: stable `pdf` target writing `output/pdf/securerag_bench_urtc_draft.pdf`.
- `scripts/verify_urtc_pdf.py`: PDF page-count and claim-boundary check.
- `tests/test_urtc_latex_pdf.py`: build and verifier regression tests.

### Task 1: Establish the LaTeX scaffold and bibliography

**Files:**
- Create: `paper/urtc/main.tex`
- Create: `paper/urtc/references.bib`
- Create: `paper/urtc/Makefile`
- Create: `tests/test_urtc_latex_pdf.py`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-31-urtc-manuscript-design.md` and `docs/claim_traceability.md`.
- Produces: `make -C paper/urtc pdf`, which writes `output/pdf/securerag_bench_urtc_draft.pdf`.

- [ ] **Step 1: Write the failing build test**

```python
def test_urtc_latex_build_creates_pdf() -> None:
    result = subprocess.run(
        ["make", "-C", "paper/urtc", "pdf"], cwd=ROOT, text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert URTC_PDF.exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_urtc_latex_pdf.py::test_urtc_latex_build_creates_pdf -v`

Expected: FAIL because the LaTeX project does not exist.

- [ ] **Step 3: Add the minimal build files**

Create `main.tex` with `\documentclass[10pt,conference]{IEEEtran}`, the packages `booktabs`, `graphicx`, `url`, and `hyperref`, plus `\bibliographystyle{IEEEtran}` and `\bibliography{references}`. Create a Makefile whose `pdf` target runs `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` and copies `main.pdf` to `../../output/pdf/securerag_bench_urtc_draft.pdf`.

Create valid, complete BibTeX records for AgentDojo, InjecAgent, CaMeL, Task Shield, and the Rubinstein-Kroese CEM book, using the verified metadata in `docs/paper_draft.md`.

- [ ] **Step 4: Run the build test to verify it passes**

Run: `python -m pytest tests/test_urtc_latex_pdf.py::test_urtc_latex_build_creates_pdf -v`

Expected: PASS and a nonempty PDF.

- [ ] **Step 5: Commit**

```bash
git add paper/urtc tests/test_urtc_latex_pdf.py
git commit -m "feat: add URTC LaTeX manuscript scaffold"
```

### Task 2: Write the claim-bounded manuscript

**Files:**
- Modify: `paper/urtc/main.tex`
- Modify: `tests/test_urtc_latex_pdf.py`

**Interfaces:**
- Consumes: Task 1 build interface plus `artifacts/fast_urtc_controlled_evaluation.json`, `artifacts/cem_extended_10.json`, `artifacts/injecagent_offline_study.json`, and `docs/claim_traceability.md`.
- Produces: a complete, claim-traceable manuscript.

- [ ] **Step 1: Add failing source-content assertions**

```python
def test_urtc_source_has_required_boundaries() -> None:
    source = (ROOT / "paper/urtc/main.tex").read_text(encoding="utf-8")
    assert "15 benign" in source
    assert "42.86\\%" in source
    assert "2,108" in source
    assert "not an official InjecAgent" in source
    assert "preliminary" in source.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_urtc_latex_pdf.py::test_urtc_source_has_required_boundaries -v`

Expected: FAIL until the approved paper content is added.

- [ ] **Step 3: Write the five-page manuscript**

Implement the approved structure:

1. Abstract: state the threat, architecture, 100% controlled utility, and reduction from 100% to 0% controlled ASR.
2. Introduction and threat model: retrieved text is attacker-controlled; distinguish retrieval manipulation from tool exfiltration.
3. Architecture: privileged planner, quarantined parser, restricted-AST interpreter, provenance labels, and capability boundary; cite CaMeL and state the implementation/evaluation contribution.
4. Evaluation: include one Roman-numbered, four-row ablation table with 15 benign tasks, seven attacks, 100%, 42.86%, and 0%; report ten CEM seeds, 100% top-5 inclusion, 0.906141 mean fitness, and 0.001968 population standard deviation.
5. Offline payload-transfer subsection: report 2,108 records, 0% transfer ASR, 100% policy halts, and immediately state that this is not an official InjecAgent score.
6. Preliminary native-pilot paragraph: report the 50-case Qwen 2.5 7B pilot's 50% valid-output rate and failed 90% gate; do not use it as a headline native score.
7. Related work, limitations, conclusion, and reproducibility: cite AgentDojo, InjecAgent, Task Shield, and CEM; retain all boundaries in `docs/claim_traceability.md`.

Use third-person prose. Captions must accompany each table or figure. Do not add appendices unless space permits.

- [ ] **Step 4: Run the source test and build**

Run:
```bash
python -m pytest tests/test_urtc_latex_pdf.py::test_urtc_source_has_required_boundaries -v
make -C paper/urtc pdf
```

Expected: PASS, no undefined citations, and a built PDF.

- [ ] **Step 5: Commit**

```bash
git add paper/urtc/main.tex tests/test_urtc_latex_pdf.py
git commit -m "docs: write URTC SecureRAG-Bench manuscript"
```

### Task 3: Add automated format and boundary checks

**Files:**
- Create: `scripts/verify_urtc_pdf.py`
- Modify: `tests/test_urtc_latex_pdf.py`

**Interfaces:**
- Consumes: `output/pdf/securerag_bench_urtc_draft.pdf`.
- Produces: `python scripts/verify_urtc_pdf.py`, which exits nonzero on a page or claim-boundary violation.

- [ ] **Step 1: Write the failing verifier test**

```python
def test_urtc_pdf_passes_verification() -> None:
    subprocess.run(["make", "-C", "paper/urtc", "pdf"], cwd=ROOT, check=True)
    result = subprocess.run(
        [sys.executable, "scripts/verify_urtc_pdf.py"], cwd=ROOT,
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pages=" in result.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_urtc_latex_pdf.py::test_urtc_pdf_passes_verification -v`

Expected: FAIL because the verifier does not exist.

- [ ] **Step 3: Implement the verifier**

Use `pypdf.PdfReader` to extract text. Fail when `len(reader.pages) > 5`, when any of `"2,108"`, `"not an official InjecAgent"`, or `"preliminary"` is missing, or when `"official InjecAgent score"` appears outside the required negated qualification. Print `pages=<count> verification=passed` on success.

- [ ] **Step 4: Run the verifier test**

Run: `python -m pytest tests/test_urtc_latex_pdf.py::test_urtc_pdf_passes_verification -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_urtc_pdf.py tests/test_urtc_latex_pdf.py
git commit -m "test: verify URTC PDF constraints"
```

### Task 4: Render and visually inspect the final draft

**Files:**
- Modify: `paper/urtc/main.tex` only if an actual layout defect or page-count violation requires it.
- Generate: `output/pdf/securerag_bench_urtc_draft.pdf`
- Generate temporarily: `tmp/pdfs/urtc-page-*.png`

**Interfaces:**
- Consumes: the build and verifier from Tasks 1-3.
- Produces: an inspected five-page-or-fewer LaTeX manuscript PDF.

- [ ] **Step 1: Build and run automated checks**

```bash
make -C paper/urtc pdf
python scripts/verify_urtc_pdf.py
python -m pytest tests/test_urtc_latex_pdf.py -v
```

Expected: all commands exit 0.

- [ ] **Step 2: Render every page**

Run: `pdftoppm -png -r 180 output/pdf/securerag_bench_urtc_draft.pdf tmp/pdfs/urtc-page`

Expected: one legible PNG for every PDF page.

- [ ] **Step 3: Inspect the rendered pages**

Check title/author area, table caption and clipping, references, section transitions, page numbers, and the preliminary-pilot qualification. Reject overlapping text, broken characters, unreadably small tables, and overflowing tables.

- [ ] **Step 4: Correct only necessary layout problems and repeat Steps 1-3**

If over five pages, remove optional prose or figure material before changing typography; do not reduce below 10-point text. If the table overflows, shorten labels or reformat the table before changing margins.

- [ ] **Step 5: Commit**

```bash
git add paper/urtc/main.tex output/pdf/securerag_bench_urtc_draft.pdf
git commit -m "docs: verify URTC manuscript PDF"
```

## Plan Self-Review

- Spec coverage: Tasks 1-2 implement the empirical paper, controlled evidence, citations, and preliminary-pilot treatment. Task 3 enforces page and wording constraints. Task 4 requires a visual PDF inspection.
- Placeholder scan: all tasks name files, commands, expected results, and fixed claim boundaries.
- Interface consistency: Task 1 defines the Make target and output path; Tasks 2-4 and the tests consume the same interface.

