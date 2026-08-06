# Claim Traceability Sheet

Use this sheet to verify every quantitative statement in the local draft before
moving it into an official conference template.

| Draft claim | Verified artifact | Exact evidence | Permitted wording |
|---|---|---|---|
| Full monitor retained utility while blocking controlled attacks | `artifacts/fast_urtc_controlled_evaluation.json` | Full monitor: 15 benign tasks, 7 attack tasks, utility 1.0, ASR 0.0, halt rate 1.0 | “In the deterministic controlled evaluation, the full monitor had 100% utility and 0% ASR.” |
| No monitoring and XML labels alone did not protect the tested flows | `artifacts/fast_urtc_controlled_evaluation.json` | Both variants: utility 1.0, ASR 1.0, halt rate 0.0 | “In this implementation, XML labels without enforcement did not block the tested flows.” |
| Provenance contributes beyond task/action policy | `artifacts/fast_urtc_controlled_evaluation.json` | Policy-only: ASR 3/7 (42.86%); full monitor: ASR 0/7 | “Removing provenance enforcement left three authorized-action taint paths successful.” |
| Trigger fragments can manipulate retrieval in the controlled setup | `artifacts/cem_extended_10.json` | 10 seeds, 30 iterations, 5,000 samples/iteration, top-5 success rate 1.0, mean fitness 0.906141, population SD 0.001968 | “Across ten deterministic CEM runs, the malicious document entered the top five in every run.” |
| The payload-transfer boundary handles diverse public payloads | `artifacts/injecagent_offline_study.json` | 2,108 cases; each suite has transfer ASR 0.0 and halt rate 1.0 | “In a generic offline payload-transfer adapter over 2,108 public InjecAgent records, transfer ASR was 0%.” |

## Prohibited shortcuts

- Do not call the InjecAgent figure an official InjecAgent score.
- Do not call the deterministic variants real LLM baselines or prompt-following
  experiments.
- Do not claim formal security for unsupported AST constructs or tools.
- Do not imply live email, payment, social-media, or browser actions occurred.

## Pre-submission check

1. Re-run the commands in `artifacts/README.md` for every table used.
2. Compare every number in the formatted manuscript against this sheet.
3. Retain the limitations paragraph when moving the draft to the official
   template.
4. Have a mentor/instructor read the threat-model and limitations claims.

## `paper/urtc/main.tex` (native validity, restricted-AST, adaptive-attack manuscript)

This section covers the **second** manuscript in this repository,
`paper/urtc/main.tex` (built by `make -C paper/urtc pdf`), which is distinct
from the offline draft covered above. Every number this manuscript reports
for the native-validity, restricted-AST, and adaptive-attack results is
generated — never hand-written — by `scripts/import_study_bundle.py` from a
**validated Kaggle study bundle** (`validate_study_bundle` in
`src/secure_rag_bench/evaluation/study_reporting.py`). Nothing in this row
group may be verified today: **`paper/generated/` currently contains only
`.gitkeep`, because no real Kaggle bundle has been imported yet.** The table
below records where each claim will come from once that import happens; it
is a traceability contract, not evidence a real run has occurred.

| Manuscript element | Generated file | `summary.json` key | Source table (bundle field) |
|---|---|---|---|
| Table `tab:native-validity`, `tab:native-execution-asr` | `paper/generated/native_validity.tex` | `summary.json["native_validity"]` | `PaperTables.native_validity` (`NativeValidityRow`) |
| Figure `fig:native-validity` | `paper/generated/native_validity_plot.tex` | `summary.json["native_validity"][*].protocol_valid_rate` | same |
| `\NativeBestValidity`, `\NativeModelCount` (abstract, evaluation, limitations, conclusion) | `paper/generated/macros.tex` | `summary.json["headline_macros"]["NativeBestValidity"/"NativeModelCount"]` | max/distinct-count over `PaperTables.native_validity` |
| Table `tab:ast-compatibility`, `tab:ast-rejection-categories` | `paper/generated/ast_compatibility.tex` | `summary.json["ast_compatibility"]` | `PaperTables.ast_compatibility` (`ASTCompatibilityRow`) |
| `\ASTBenignUtility` (abstract, evaluation, conclusion) | `paper/generated/macros.tex` | `summary.json["headline_macros"]["ASTBenignUtility"]` | min `execution_rate` over `PaperTables.ast_compatibility` |
| Table `tab:adaptive-monitor` | `paper/generated/adaptive_monitor.tex` | `summary.json["adaptive_monitor"]` | `PaperTables.adaptive_monitor` (`AdaptiveMonitorRow`) |
| Figure `fig:adaptive-results` | `paper/generated/adaptive_results_plot.tex` | `summary.json["adaptive_monitor"][*].target_effect_asr` | same, `monitor == "full_monitor"` rows |
| `\AdaptiveFullASR` (abstract, evaluation, conclusion) | `paper/generated/macros.tex` | `summary.json["headline_macros"]["AdaptiveFullASR"]` | max `target_effect_asr` over `full_monitor` rows |
| File-integrity check for every file above | `paper/generated/MANIFEST.sha256` | n/a (SHA-256 per file, not a JSON key) | covers every file `_write_outputs` writes, including `macros.tex` and both plot files |

**Verification procedure once a real bundle exists:**

1. Run `python scripts/import_study_bundle.py <real-kaggle-bundle>.zip --output-dir paper/generated`.
2. Confirm `sha256sum -c paper/generated/MANIFEST.sha256` (or equivalent)
   passes for every listed file.
3. Confirm every number quoted in `main.tex` via a `\newcommand` macro or a
   `\input`-ed table equals the corresponding `summary.json` value above,
   rounded the same way `scripts/import_study_bundle.py` rounds it
   (`_pct`/`_format_macro_value`: two decimal places).
4. Re-run `python -m pytest tests/test_urtc_latex_pdf.py tests/test_study_bundle_scripts.py -v`.
5. Only then may `docs/submission_readiness.md`'s "paper/urtc" status move
   from BLOCKED.

## Prohibited shortcuts (paper/urtc)

- Do not hand-write a native-validity, restricted-AST, or adaptive-attack
  number into `main.tex`; every new numeric claim must come from a
  `\newcommand` macro in `paper/generated/macros.tex` or a `\input`-ed table.
- Do not commit any file under `paper/generated/` other than `.gitkeep`
  until a real Kaggle bundle has been imported; fixture/synthetic bundle
  outputs used in tests are confined to `tmp_path`/`tmp/` and are never
  copied into `paper/generated/`.
- Do not present a skipped or excluded model's non-result as if it were a
  measured outcome; `build_paper_tables` already excludes any configuration
  whose `status` is not `"completed"`.
