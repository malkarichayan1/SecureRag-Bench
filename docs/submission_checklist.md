# MIT URTC Submission Checklist

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
- [ ] Refresh the draft's older six-benign/two-attack and three-seed numbers using `docs/experimental_evidence_brief.md` when paper drafting begins.
- [ ] Add citations and format to the official URTC template.
- [ ] Add author information and advisor/mentor acknowledgement if applicable.
- [ ] Obtain a technical review from a mentor or instructor.
- [x] Produce a current review-ready offline PDF draft at `output/pdf/securerag_bench_offline_draft.pdf`.
- [ ] Inspect the PDF visually after confirming the official URTC template.

## Before submission

- [ ] Re-run the full test suite.
- [ ] Re-run commands used for every table.
- [ ] Verify repository and artifact instructions from a clean environment.
- [ ] Upload before the portal deadline; do not wait for the final hour.
