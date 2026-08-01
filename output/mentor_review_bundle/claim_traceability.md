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
