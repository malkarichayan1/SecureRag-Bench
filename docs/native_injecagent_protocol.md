# Native InjecAgent Evaluation Protocol

## Scope

This protocol measures local-model vulnerability and execution gating with the
official InjecAgent prompted-agent parser and score definitions. It does not
claim that native InjecAgent isolates provenance enforcement: a successful
native attack selects an attacker tool instead of the user tool, so a
task-alignment gate blocks that event by construction.

## Research questions

1. How often does the unprotected local model propose a benchmark-defined
   attacker action?
2. How often would that proposed action be permitted by a task-alignment gate?
3. Does the existing authorized-action replay study show the separate value of
   provenance when the action itself is permitted but an argument is tainted?

## Conditions

Native runs use exactly two conditions:

- `no_defense`: proposed attacker actions are permitted.
- `task_alignment_guard`: a proposed tool must equal the original user tool.

The runner saves the model's official score in `scores` and the score after
the execution gate in `execution_scores`. These answer different questions and
must never be substituted for one another.

For data-stealing cases, a denied first-step attacker action intentionally
prevents a second model interaction. Consequently, use the `no_defense` run as
the complete native model-vulnerability measurement and use guarded artifacts
for execution ASR; do not compare their model-level total ASRs as if they were
matched completed trajectories.

## Fixed experiment configuration

- Primary model: `Qwen/Qwen2.5-7B-Instruct`, with the exact Hugging Face
  revision recorded before the first run.
- Settings: `base` and `enhanced`.
- Prompt type: `InjecAgent`.
- Decoding: greedy (`do_sample=False`) and `max_new_tokens=512`.
- Dataset: the pinned `data/external/InjecAgent` submodule revision.
- Side effects: none. The runner records a simulated execution decision only.

## Pilot gate

Run 25 direct-harm and 25 data-stealing cases per setting before the full
matrix. Inspect at least 20 model outputs across both attack classes. Do not
launch the full study if valid rate is below 90%, an evaluator disagreement is
unexplained, or a saved record cannot be traced to its model output and gate
decision.

## Full matrix

For each setting, run both conditions over all 1,054 cases and retain one JSON
artifact per condition. This produces four primary artifacts. Run a second
model family only after the primary matrix passes the pilot gate.

## Provenance-specific evidence

Use `artifacts/injecagent_free_baselines.json` and
`artifacts/fast_urtc_controlled_evaluation.json` for the authorized-action
replay claim. In that evaluation, the action may be authorized while the data
argument is tool-sourced; policy-only and full-monitor results therefore
isolate the contribution of provenance enforcement more directly than native
InjecAgent does.

## Reporting

Report all of the following:

- Native ASR-valid and ASR-all for model proposals.
- Native ASR-valid and ASR-all after the task-alignment gate.
- Invalid-output count and a manual error taxonomy.
- Wilson 95% intervals for model and execution ASR-all.
- The offline authorized-action replay result separately, with its existing
  limitations.

Never label the offline replay as an official InjecAgent score or label the
native task-alignment gate as a provenance ablation.
