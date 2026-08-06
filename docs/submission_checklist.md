# MIT URTC Submission Checklist

This repository carries **two** manuscripts. The sections below through
"Before submission" track the offline/deterministic draft
(`docs/paper_draft.md`, `output/pdf/securerag_bench_offline_draft.pdf`). The
final section, "Native validity / adaptive-attack manuscript
(`paper/urtc/`)", tracks the separate `paper/urtc/main.tex` manuscript, which
is not ready for any of the steps below until it is unblocked -- see that
section and `docs/submission_readiness.md`.

## Deadline and eligibility

- [ ] Confirm the 2026 technical-paper deadline in the official submission
  portal or directly with MIT URTC organizers. The public site currently shows
  a stale 2025 deadline alongside 2026 event dates.
- [ ] Confirm paper length, template, anonymity, authorship, and undergraduate
  eligibility requirements for the 2026 call for papers.

## Evidence package

- [x] Restricted-AST interpreter and provenance tests.
- [x] Four-configuration controlled evaluation artifact (15 benign tasks; seven attack paths).
- [x] Complete ten CEM seeds and save aggregate mean and deviation.
- [x] Complete the 2,108-case InjecAgent offline payload-transfer artifact with stratified output.
- [x] Check primary reported numbers against saved JSON artifacts.
- [x] Maintain a claim-to-artifact traceability sheet at `docs/claim_traceability.md`.
- [x] Record SHA-256 digests for primary paper-facing artifacts in `artifacts/MANIFEST.sha256`.

## Paper package

- [x] Initial abstract, method, results, limitations, and reproducibility draft.
- [x] Refresh the draft with the 15-benign/seven-attack controlled results, ten-seed CEM study, and 2,108-case offline transfer study.
- [ ] Add citations and format to the official URTC template.
- [ ] Add author information and advisor/mentor acknowledgement if applicable.
- [ ] Obtain a technical review from a mentor or instructor.
- [x] Produce a current review-ready offline PDF draft at `output/pdf/securerag_bench_offline_draft.pdf`.
- [ ] Inspect the PDF visually after confirming the official URTC template.

## Before submission

- [x] Re-run the full test suite (clean-environment run: 48 tests passed on 2026-07-28).
- [x] Re-run the commands used for paper tables and verify primary artifact hashes (2026-07-28).
- [x] Verify repository installation, tests, and local PDF generation from a clean temporary environment (2026-07-28).
- [ ] Upload before the portal deadline; do not wait for the final hour.

## Native validity / adaptive-attack manuscript (`paper/urtc/`)

This is a second, separate manuscript from the offline draft above (see
`docs/submission_readiness.md`'s "`paper/urtc/main.tex` native-validity
manuscript: status BLOCKED" section for the authoritative status). Every
item below is honestly **not yet done** -- no real Kaggle bundle has been
produced or imported as of this writing, and none of these steps can be
checked off until one has:

- [ ] Run `notebooks/securerag_native_adaptive_kaggle.ipynb` end-to-end on a
  Kaggle GPU instance for at least one qualifying model (README.md's "Kaggle
  native validity and adaptive attack study" section documents the full
  workflow: GPU/HF-token setup, the 90% held-out validity gate, checkpoint
  resumability, and bundle export).
- [ ] Download the exported study bundle and import it locally with
  `python scripts/import_study_bundle.py <bundle>.zip --output-dir
  paper/generated`, confirming it passes `validate_study_bundle` and that
  `paper/generated/` now contains real (not `.gitkeep`-only) generated
  tables, plot data, and `macros.tex`.
- [ ] Build `paper/urtc/main.tex` with `make -C paper/urtc pdf` and run
  `python scripts/verify_urtc_pdf.py
  output/pdf/securerag_bench_urtc_draft.pdf`.
- [ ] Visually inspect every rendered PDF page (clipped text, overlapping
  tables, broken references, unreadable plots, page overflow).
- [ ] Check every quantitative claim in the built manuscript against
  `docs/claim_traceability.md`'s `paper/urtc/main.tex` table.
- [ ] Only after all of the above, update `docs/submission_readiness.md` to
  move this manuscript's status off BLOCKED.
- [ ] Confirm 2026 MIT URTC deadline, template, anonymity, authorship, and
  eligibility requirements apply the same way to this manuscript as to the
  offline draft above before treating it as submission-ready.
