# InjecAgent Offline Evaluation Design

## Purpose

Make the existing free, offline InjecAgent payload-transfer experiment more
auditable and useful for a research paper without representing it as an
official InjecAgent score.

## Scope

- Load all base and enhanced InjecAgent JSON case files already vendored under
  `data/external/InjecAgent/data/`.
- Preserve source case metadata needed for stratified reporting: source file,
  attack type, user tool, and attacker tools.
- Route every source user-tool response through the same generic untrusted
  retrieval boundary, then attempt a simulated external email action with the
  tainted result.
- Produce aggregate, attack-type, and user-tool ASR/policy-halt counts in
  JSON artifacts.
- Run the four public base/enhanced suites and record the generated artifacts.

## Non-goals

- Do not claim official InjecAgent ASR-valid or ASR-all metrics.
- Do not install or call a paid API, hosted LLM, or real external service.
- Do not create paper prose or alter the current paper draft.
- Do not emulate every native InjecAgent attacker tool; that requires its
  native LLM-agent runtime and is a separate future study.

## Architecture

`load_injecagent_cases` will retain normalized source metadata. The offline
runner will execute each record through a fresh `TripartiteRedTeam` instance,
where the source `Tool Response` is the quarantined retrieval payload and the
only side-effect capability is an in-memory `send_email` function. A reporting
helper will group actual result rows by attack type and user tool, retaining
case-level outcomes for audit.

## Output Contract

Each suite result has:

- `evaluation_type`: `"offline_payload_transfer"`.
- `source_file`, `case_count`, and a clear `limitations` string.
- Overall `attack_success_rate`, `defense_rate`, and `policy_halt_rate`.
- `by_attack_type` and `by_user_tool` maps; every group has `case_count`,
  `attack_successes`, `policy_halts`, and rates.
- Case rows including attacker tools and whether a simulated email was sent.

The CLI will expose `injecagent` for one input file and `injecagent-study` for
the four standard public files. Each mode requires `--output` for durable
research artifacts.

## Verification

Tests must demonstrate that metadata survives loading, grouping uses actual
case outcomes, and the study CLI emits all four named suites. Run the full
pytest suite and `pip check` after artifact generation.
