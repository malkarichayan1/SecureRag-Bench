# SecureRAG-Bench Mentor Review Bundle

This local bundle packages the current offline research draft and the exact
evidence needed to review it. It is not a conference submission package and
contains no external upload configuration.

## Start here

1. Read `paper_draft.md` and `securerag_bench_offline_draft.pdf`.
2. Check every numeric statement against `claim_traceability.md`.
3. Review the design limits in `experimental_evidence_brief.md`.
4. Use `submission_checklist.md` to identify the remaining external tasks.

## Result artifacts

- `fast_urtc_controlled_evaluation.json`: four-configuration, 15-benign/seven-attack controlled evaluation.
- `cem_extended_10.json`: ten full seeded CEM runs.
- `injecagent_offline_study.json`: 2,108-case generic offline payload-transfer study.
- `MANIFEST.sha256`: integrity hashes for the three result artifacts.

## Important limitation

The InjecAgent study is a generic offline payload-transfer adapter, not an
official InjecAgent score. The model components and external capabilities are
deterministic and simulated. Keep these limitations in any formatted version.
