# Free InjecAgent Baselines Design

## Purpose

Add a reproducible, fully offline comparison of SecureRAG-Bench's reference
monitor against transparent baseline defense families on the vendored InjecAgent
payloads. This study is a payload-transfer evaluation, not an official
InjecAgent, AgentDojo, PromptArmor, or Task Shield score.

## Scope

- Evaluate all four vendored InjecAgent suites through the existing generic
  untrusted-retrieval and simulated-external-email boundary.
- Compare six variants: no defense, XML delimiters, keyword sanitizer,
  task-alignment guard, policy-only monitor, and full provenance-plus-policy
  monitor.
- Report per-suite and aggregate attack success rate, defense rate, policy-halt
  rate, and controlled benign-task utility.
- Store the selected implementation variant and case-level outcomes in JSON.

## Baseline Definitions

`no_defense` disables the existing policy and provenance checks.

`xml_delimiters_only` wraps retrieval text in explicit untrusted-data tags but
does not inspect it or constrain an external action.

`keyword_sanitizer` removes instruction-like lines containing an imperative
override pattern before the parser receives retrieval text. It is a
deterministic sanitizer surrogate, not a reproduction of the LLM-based
PromptArmor method.

`task_alignment_guard` permits a requested external action only when its tool,
recipient, and required content are directly supported by the user task. It is
a deterministic task-alignment surrogate, not a reproduction of Task Shield.

`policy_only` preserves contextual policy checks while disabling provenance
enforcement. `full_monitor` uses the existing provenance and contextual-policy
enforcement.

## Architecture

Add a small baseline-configuration layer that builds a fresh evaluation target
for each variant. All targets receive the exact same normalized case, tool
response, attack plan, and simulated email capability. A report function will
run every variant over every suite, then summarize comparable metrics by suite,
variant, attack type, and source user tool.

The existing offline study remains unchanged for backwards compatibility. A
new CLI mode produces a separate artifact and requires no network, model API,
or extra dependency.

## Error Handling

The runner will reject unknown baseline names and missing standard suite files.
It will retain per-case error details rather than counting evaluation failures
as blocked attacks.

## Verification

Unit tests will establish each baseline's distinctive behavior using one
tainted payload and one benign request. CLI tests will verify all four suites
and all six variants are emitted. The full test suite and dependency check will
run before artifacts are generated.

## Reporting Limits

Artifacts and documentation must state that the sanitizer and alignment guard
are deterministic baseline families inspired by published defenses. They must
not attribute the resulting scores to PromptArmor, Task Shield, AgentDojo, or
the official InjecAgent protocol.
