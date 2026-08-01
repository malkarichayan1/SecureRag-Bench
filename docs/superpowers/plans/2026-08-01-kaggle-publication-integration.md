# Kaggle and Publication Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a resumable Kaggle notebook, validated result bundle, generated paper tables, and a URTC manuscript source that cannot claim unexecuted experiments.

**Architecture:** Tested repository APIs own all generation, scoring, replay, adaptive evaluation, and aggregation. A generated notebook only orchestrates those APIs; an import tool validates the exported bundle before producing deterministic CSV/LaTeX tables consumed by the manuscript.

**Tech Stack:** Python 3.10+, nbformat/Jupyter, JSON/JSONL/SHA-256, matplotlib, LaTeX/IEEEtran, pytest, Poppler/PyPDF for verification.

## Global Constraints

- Run after both `2026-08-01-native-validity-pipeline.md` and `2026-08-01-adaptive-ast-evaluation.md`.
- The user executes Kaggle; local tests use fake adapters and fixture artifacts only.
- Notebook secrets must never enter outputs, exceptions, prompts, logs, or bundles.
- The full native stage must refuse configurations below 90% held-out protocol validity.
- Skipped/unexecuted optional models must be excluded from empirical tables.
- Manuscript results must be generated only from a validated bundle.
- Final PDF must remain within five pages and must be rendered and visually inspected.

---

## File Structure

- `src/secure_rag_bench/evaluation/study_reporting.py`: bundle validation, cross-artifact consistency, table rows, and plot data.
- `scripts/export_study_bundle.py`: Kaggle-side packaging and manifest generation.
- `scripts/import_study_bundle.py`: local validation and deterministic CSV/LaTeX generation.
- `scripts/build_kaggle_notebook.py`: reproducible notebook builder.
- `notebooks/securerag_native_adaptive_kaggle.ipynb`: generated user-run notebook.
- `paper/urtc/`: integrated IEEE manuscript source from `codex/urtc-latex-manuscript` plus generated result inputs.
- `tests/test_study_reporting.py`, `tests/test_kaggle_notebook.py`, `tests/test_study_bundle_scripts.py`, `tests/test_urtc_latex_pdf.py`: reporting, orchestration, and publication verification.

### Task 1: Validate and Aggregate the Complete Study Bundle

**Files:**
- Create: `src/secure_rag_bench/evaluation/study_reporting.py`
- Create: `tests/test_study_reporting.py`

**Interfaces:**
- Consumes raw native JSONL, split manifest, gate decisions, replay records, AST records, adaptive records, model/environment metadata, and SHA-256 manifest.
- Produces: `validate_study_bundle(root: Path) -> ValidatedStudyBundle`, `build_paper_tables(bundle) -> PaperTables`, and `build_plot_series(bundle) -> dict[str, Any]`.

- [ ] **Step 1: Write failing bundle rejection tests**

```python
def test_bundle_rejects_hash_mismatch_missing_case_and_failed_gate_claim(tmp_path) -> None:
    bundle = fixture_bundle(tmp_path, qualifying=True)
    corrupt_one_record(bundle)
    with pytest.raises(BundleValidationError, match="sha256"):
        validate_study_bundle(bundle.root)

def test_unexecuted_optional_model_is_absent_from_table_rows(tmp_path) -> None:
    bundle = validate_study_bundle(fixture_bundle(tmp_path, skipped_optional=True).root)
    tables = build_paper_tables(bundle)
    assert "claude" not in {row.model for row in tables.native_validity}
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/test_study_reporting.py -v`

- [ ] **Step 3: Implement cross-artifact validation**

Validate hashes, unique IDs, expected split membership, 25/25 pilot balance, gate arithmetic, full-run case coverage, model/dataset revision presence, raw-to-summary denominators, replay source hashes, AST/adaptive schema versions, and absence of secret-shaped keys/values.

- [ ] **Step 4: Implement typed paper tables**

```python
@dataclass(frozen=True)
class PaperTables:
    native_validity: tuple[NativeValidityRow, ...]
    adaptive_monitor: tuple[AdaptiveMonitorRow, ...]
    ast_compatibility: tuple[ASTCompatibilityRow, ...]
```

Each row carries counts as well as rates and intervals. Only `status == "completed"` configurations with valid manifests can be emitted.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_study_reporting.py -v`

```bash
git add src/secure_rag_bench/evaluation/study_reporting.py tests/test_study_reporting.py
git commit -m "feat: validate complete study result bundles"
```

### Task 2: Export and Import Scripts with Deterministic Paper Inputs

**Files:**
- Create: `scripts/export_study_bundle.py`
- Create: `scripts/import_study_bundle.py`
- Create: `tests/test_study_bundle_scripts.py`
- Create: `paper/generated/.gitkeep`

**Interfaces:**
- CLI export: `python scripts/export_study_bundle.py --run-root PATH --output PATH.zip`.
- CLI import: `python scripts/import_study_bundle.py BUNDLE.zip --output-dir paper/generated`.
- Produces: `native_validity.tex`, `adaptive_monitor.tex`, `ast_compatibility.tex`, matching CSV files, `summary.json`, and `MANIFEST.sha256`.

- [ ] **Step 1: Write failing round-trip test**

```python
def test_export_import_round_trip_generates_only_validated_rows(tmp_path) -> None:
    archive = export_fixture_bundle(tmp_path)
    result = run_import(archive, tmp_path / "paper-generated")
    assert result.exit_code == 0
    assert (result.output / "native_validity.tex").exists()
    assert "optional-skipped" not in (result.output / "native_validity.tex").read_text()
    assert verify_manifest(result.output / "MANIFEST.sha256")
```

- [ ] **Step 2: Run script tests and confirm failure**

Run: `python -m pytest tests/test_study_bundle_scripts.py -v`

- [ ] **Step 3: Implement safe archive export/import**

Reject absolute paths, `..` traversal, symlinks, duplicate members, files outside the expected schema, and hash mismatches. Import into a temporary directory, validate through `validate_study_bundle`, generate files, then atomically replace the output directory.

- [ ] **Step 4: Generate escaped LaTeX and machine-readable CSV**

Use one escaping function for `\`, `%`, `_`, `&`, `#`, `{`, and `}`. Format rates with declared denominators and confidence intervals; never parse rounded display strings back into numbers.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_study_bundle_scripts.py tests/test_study_reporting.py -v`

```bash
git add scripts/export_study_bundle.py scripts/import_study_bundle.py tests/test_study_bundle_scripts.py paper/generated/.gitkeep
git commit -m "feat: export validated paper result inputs"
```

### Task 3: Build the Resumable Kaggle Notebook

**Files:**
- Create: `scripts/build_kaggle_notebook.py`
- Create: `notebooks/securerag_native_adaptive_kaggle.ipynb`
- Create: `tests/test_kaggle_notebook.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Builder: `build_notebook(output: Path) -> None`.
- Notebook stages: `preflight`, `pilot`, `full_native`, `ast_adaptive`, and `export`.

- [ ] **Step 1: Write failing notebook-structure test**

```python
def test_notebook_has_ordered_guarded_stages_and_no_embedded_secrets() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    sources = "\n".join(cell.source for cell in notebook.cells)
    assert [cell.metadata.get("stage") for cell in notebook.cells if cell.metadata.get("stage")] == [
        "preflight", "pilot", "full_native", "ast_adaptive", "export"
    ]
    assert "evaluate_validity_gate" in sources
    assert "if not gate.passed" in sources
    assert "sk-" not in sources and "ANTHROPIC_API_KEY=" not in sources
```

- [ ] **Step 2: Run notebook test and confirm failure**

Run: `python -m pytest tests/test_kaggle_notebook.py -v`

- [ ] **Step 3: Implement a deterministic notebook builder**

Use `nbformat` to create markdown instructions and small code cells that clone/upload the repo, install `.[local-injecagent]`, mount a persistent Kaggle output directory, inspect GPU/disk/model access, serialize the split, run pilots, display gate summaries, run only qualifying full configurations, execute AST/adaptive stages, and export the bundle.

- [ ] **Step 4: Add explicit safety and resumption cells**

The preflight cell must print only credential presence booleans, not values. Every run cell must use `JsonlCheckpointStore`; the full-native cell must raise `RuntimeError("validity gate not passed")` before model loading when the selected pilot failed.

- [ ] **Step 5: Build and statically execute CPU-only cells**

Run: `python scripts/build_kaggle_notebook.py`

Run: `python -m pytest tests/test_kaggle_notebook.py -v`

Use a test harness to execute only cells tagged `cpu_smoke`, with fake adapters and temporary paths.

- [ ] **Step 6: Add notebook dependency and commit**

Add `nbformat>=5.10` to the development dependency group.

```bash
git add pyproject.toml scripts/build_kaggle_notebook.py notebooks/securerag_native_adaptive_kaggle.ipynb tests/test_kaggle_notebook.py
git commit -m "feat: add resumable kaggle study notebook"
```

### Task 4: Integrate the Existing URTC LaTeX Source

**Files:**
- Create from branch: `paper/urtc/Makefile`
- Create from branch: `paper/urtc/main.tex`
- Create from branch: `paper/urtc/references.bib`
- Create from branch: `paper/urtc/figures/architecture.tex`
- Create from branch: `paper/urtc/figures/ablation_results.tex`
- Create from branch: `scripts/verify_urtc_pdf.py`
- Create from branch: `tests/test_urtc_latex_pdf.py`

**Interfaces:**
- Consumes the already-reviewed files committed on `codex/urtc-latex-manuscript`.
- Produces a compilable IEEEtran manuscript baseline on the implementation branch.

- [ ] **Step 1: Import only the manuscript paths from the existing branch**

```bash
git restore --source codex/urtc-latex-manuscript -- paper/urtc scripts/verify_urtc_pdf.py tests/test_urtc_latex_pdf.py
```

Do not merge or restore branch versions of `artifacts/`, `docs/`, `src/`, or existing output bundles because `main` contains newer pilot evidence and the approved study design.

- [ ] **Step 2: Run source-level manuscript tests**

Run: `python -m pytest tests/test_urtc_latex_pdf.py -v`

Expected: source checks pass; PDF-dependent checks skip only if LaTeX is unavailable.

- [ ] **Step 3: Build and verify when LaTeX is installed**

Run: `make -C paper/urtc pdf`

Run: `python scripts/verify_urtc_pdf.py output/pdf/securerag_bench_urtc_draft.pdf`

Expected: successful build, <= 5 pages, embedded text includes required headings/references, and no missing citations.

- [ ] **Step 4: Commit the isolated manuscript import**

```bash
git add paper/urtc scripts/verify_urtc_pdf.py tests/test_urtc_latex_pdf.py
git commit -m "docs: integrate urtc latex manuscript source"
```

### Task 5: Revise the Manuscript to Consume Validated Results

**Files:**
- Modify: `paper/urtc/main.tex`
- Modify: `paper/urtc/Makefile`
- Create: `paper/urtc/figures/native_validity.tex`
- Create: `paper/urtc/figures/adaptive_results.tex`
- Modify: `tests/test_urtc_latex_pdf.py`
- Modify: `docs/claim_traceability.md`
- Modify: `docs/submission_readiness.md`

**Interfaces:**
- Consumes `paper/generated/*.tex` from Task 2.
- Produces a manuscript that includes only completed tables and traces every numeric claim to `summary.json` and its manifest.

- [ ] **Step 1: Write failing source guards**

```python
def test_manuscript_requires_generated_validated_tables() -> None:
    source = MAIN_TEX.read_text(encoding="utf-8")
    assert r"\input{../../generated/native_validity.tex}" in source
    assert "50\\% of outputs were valid" not in source
    assert "Claude" not in source or "optional" in source

def test_no_anticipated_result_language() -> None:
    source = MAIN_TEX.read_text(encoding="utf-8").lower()
    assert "expected to achieve" not in source
    assert "we anticipate" not in source
```

- [ ] **Step 2: Revise evaluation structure without inventing results**

Replace the preliminary-pilot paragraph with methods and an input of validated native validity/results. Add native validity diagnosis, qualifying full study, restricted-AST compatibility, and integrated adaptive attacks. Compress the generic payload-transfer paragraph and controlled chart. If a validated table is absent, the build must stop with a clear message rather than insert substitute result text.

- [ ] **Step 3: Update abstract/conclusion through generated macros**

Generate exact headline macros such as `\NativeBestValidity`, `\NativeModelCount`, `\AdaptiveFullASR`, and `\ASTBenignUtility` during import. The manuscript may use only those macros for new numerical claims.

- [ ] **Step 4: Update traceability and readiness**

Map each table, figure, abstract statistic, and conclusion statement to the bundle file, summary key, and SHA-256 manifest. Mark submission blocked until the user's real Kaggle bundle validates and the rendered PDF passes review.

- [ ] **Step 5: Run source and fixture-PDF tests**

Run: `python -m pytest tests/test_urtc_latex_pdf.py tests/test_paper_pdf.py -v`

Use a validated fixture bundle to exercise compilation without presenting fixture values as real results; fixture outputs remain under `tmp/` and are never committed to `paper/generated`.

- [ ] **Step 6: Commit**

```bash
git add paper/urtc docs/claim_traceability.md docs/submission_readiness.md tests/test_urtc_latex_pdf.py
git commit -m "docs: wire urtc manuscript to validated study results"
```

### Task 6: End-to-End Verification and User Execution Handoff

**Files:**
- Modify: `README.md`
- Modify: `artifacts/README.md`
- Modify: `docs/submission_checklist.md`
- Modify: `tests/test_package_smoke.py`

**Interfaces:**
- Produces exact local smoke, Kaggle execution, bundle download/import, manuscript build, and final verification instructions.

- [ ] **Step 1: Add a failing documentation smoke test**

```python
def test_readme_documents_kaggle_gate_resume_and_bundle_import() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for phrase in ("90% validity gate", "resume", "export_study_bundle.py", "import_study_bundle.py"):
        assert phrase in readme
```

- [ ] **Step 2: Document the exact user workflow**

Include notebook upload/open, GPU enablement, Hugging Face license/token setup, optional credentials, preflight interpretation, pilot gate review, full run, checkpoint download, bundle export, local import, paper build, and the rule that optional/skipped models are excluded.

- [ ] **Step 3: Run complete verification**

Run: `python -m pytest -q`

Run: `python -m pip check`

Run: `python scripts/build_kaggle_notebook.py`

Run: `git diff --check`

Expected: all tests pass, dependencies are consistent, notebook regeneration is clean, and no whitespace errors exist.

- [ ] **Step 4: Build, render, and visually inspect the final fixture manuscript**

Run: `make -C paper/urtc pdf`

Run: `python scripts/verify_urtc_pdf.py output/pdf/securerag_bench_urtc_draft.pdf`

Render all PDF pages with Poppler and inspect for clipped text, overlapping tables, broken references, unreadable plots, and page overflow. Do not claim the real empirical paper is complete until the user's Kaggle bundle replaces fixture inputs and the same checks pass again.

- [ ] **Step 5: Commit the handoff documentation**

```bash
git add README.md artifacts/README.md docs/submission_checklist.md tests/test_package_smoke.py notebooks/securerag_native_adaptive_kaggle.ipynb
git commit -m "docs: add kaggle study execution handoff"
```
