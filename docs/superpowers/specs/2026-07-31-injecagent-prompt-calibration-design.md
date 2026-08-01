# InjecAgent Prompt-Adherence Calibration Design

## Context

The Qwen2.5-7B base pilot used the official InjecAgent parser and scorer over
25 direct-harm and 25 data-stealing cases. It produced a 50.0% valid rate:
10 first-stage successes, 15 unsuccessful outputs, and 25 invalid outputs.
The invalid records include both formatting/trajectory failures and model
behavior failures, so the full 1,054-case matrix is currently gated.

## Goal

Improve protocol adherence through the project-owned prompt wrapper while
preserving the official parser, score definitions, defense gate, greedy
decoding, model revision, case selection, and pilot size. The calibration is
successful only if it materially reduces format-only invalid outputs without
masking behavioral failures.

## Non-goals

- Do not modify the vendored InjecAgent parser or scorer.
- Do not reclassify invalid records as successful or unsuccessful.
- Do not add constrained decoding, sampling, or model-specific stopping rules.
- Do not run the full matrix until the 90% valid-rate pilot gate is met.

## Design

Add explicit instructions to the project-owned prompt wrapper that the model
must emit exactly one allowed tool action with an action input, or a final
answer without an `Action: Final Answer` line. Keep the official available-tool
list and scratchpad unchanged. The wrapper remains responsible only for
prompt construction; parsing and scoring remain downstream and unchanged.

The pilot reruns the same 50 base-setting cases with Qwen2.5-7B-Instruct at
the recorded revision, greedy decoding, and both `no_defense` and
`task_alignment_guard` conditions. Results must be saved as separate,
self-describing JSON artifacts with `only_first_step` recorded in `protocol`.

## Testing

- Add a unit test showing the calibrated prompt contains the required output
  contract and does not expose scorer internals.
- Preserve existing prompt, parser, native-score, execution-gate, and artifact
  persistence tests.
- Run the full test suite before launching Kaggle.
- Inspect at least 20 pilot outputs across both attack classes and retain an
  invalid-output taxonomy.

## Acceptance criteria

1. All tests pass.
2. Both pilot artifacts contain 50 cases, 25 per attack, zero runner errors,
   the model revision, and `only_first_step: true`.
3. The calibrated valid rate is reported alongside the prior 50.0% result.
4. The official parser/scorer code and defense semantics are unchanged.
5. The full matrix remains blocked unless valid rate reaches the documented
   90% threshold and the output review finds no unexplained evaluator issue.
