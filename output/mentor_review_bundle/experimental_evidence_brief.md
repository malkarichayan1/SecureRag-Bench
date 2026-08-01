# Experimental Evidence Brief

This drafting aid maps verified artifacts to paper sections. It is not a paper draft.

## Threat model

The controlled attacker controls retrieved or tool-returned text and attempts to route it into an external side effect. Tested paths are email body, subject, recipient, unauthorized email, transfer amount, and external-post body. The CEM artifact separately demonstrates retrieval-rank manipulation.

## Architecture

The target separates privileged planning from quarantined parsing, executes only restricted AST code, tracks provenance, and checks each tool action with the reference monitor. XML tags label untrusted content but are not the enforcement mechanism.

## Security invariant

Within the implemented policy, untrusted tool-derived values cannot reach an external side-effect capability when provenance enforcement is enabled. This claim applies to supported restricted-AST operations and configured tools.

## Evaluation evidence

| Artifact | What it establishes |
|---|---|
| `artifacts/fast_urtc_controlled_evaluation.json` | 15 benign tasks; seven controlled attack paths; four configurations. |
| `artifacts/cem_extended_10.json` | Ten full seeded trigger-fragment runs; malicious document reached top-5 in every run. |
| `artifacts/injecagent_offline_study.json` | 2,108 public benchmark payloads in the generic offline transfer adapter. |

## Controlled comparison result

| Configuration | Utility | ASR | Policy-halt rate |
|---|---:|---:|---:|
| No monitor | 100% | 100% | 0% |
| XML delimiters only | 100% | 100% | 0% |
| Policy only (no provenance) | 100% | 42.86% | 57.14% |
| Full monitor | 100% | 0% | 100% |

Interpretation: XML labels alone did not prevent the tested flows; action/task policy blocked four paths; provenance enforcement blocked the three remaining authorized-action taint flows.

## Limitations to retain in the paper

- Planner, attacker, and jury are deterministic mocks; no prompt-following or real-model robustness claim is supported.
- The InjecAgent adapter maps source tools to generic retrieval and simulated email, so it is not an official ASR-valid or ASR-all score.
- The policy covers an explicit restricted AST and configured tool set, not arbitrary agent implementations.
- CEM results use the supplied deterministic embedding/retrieval setup.

## Regeneration

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval ablation --output artifacts\fast_urtc_controlled_evaluation.json
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval injecagent-study --injecagent-dir data\external\InjecAgent\data --output artifacts\injecagent_offline_study.json
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval study --seeds 1 2 3 4 5 6 7 8 9 10 --output artifacts\cem_extended_10.json
```
