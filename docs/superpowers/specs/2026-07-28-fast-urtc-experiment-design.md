# Fast Defensible URTC Experiment Design

## Goal

Produce a paper-ready controlled evaluation of architectural prompt-injection
defenses without paid APIs, local-model installation, or claims that a
deterministic mock measures prompt-following behavior.

## Research questions

1. Does a reference monitor prevent the tested indirect prompt-injection flows
   while retaining benign task utility?
2. Does provenance enforcement add protection beyond task/action policy?
3. Do XML delimiters alone prevent attacks when no enforcement mechanism is
   present?
4. Does the defense generalize across message body, subject, recipient,
   transfer, and external-post action paths?

## Systems compared

- `no_monitor`: a naive agent with no policy or provenance checks.
- `xml_delimiters_only`: identical execution permissions to `no_monitor`, but
  untrusted retrieval remains XML-delimited. This isolates labels from
  enforcement and is not presented as a prompt-engineering baseline.
- `policy_only`: task/action checks active; provenance checks disabled.
- `no_provenance_enforcement`: retained legacy name for `policy_only` output,
  to preserve existing artifacts.
- `full_monitor`: task, source authorization, action alignment, and data
  isolation active.

No defensive-prompt baseline will be claimed. A deterministic planner cannot
measure whether a language model follows an instruction; adding one would make
the comparison misleading. Real-model prompt baselines remain future work.

## Workload

- Keep the 12 existing benign tasks and add multi-step trusted workflow tasks
  for retrieve/format, trusted email, and trusted transfer paths.
- Add attack vectors for tainted transfer amount and tainted external-post
  content, alongside existing body, subject, recipient, and unauthorized
  external-action vectors.
- Report utility, ASR, defense rate, halt rate, and per-vector outcomes for
  every configuration.
- Retain the CEM study and 2,108-case InjecAgent offline transfer study as
  separate retrieval and scale evidence.

## Results contract

The consolidated artifact must contain a `scope` object that declares this is
a deterministic controlled evaluation, the exact system variants, counts, and
per-vector outcomes. It will include no native benchmark or real-model claim.

## Paper handoff

After verification, add a concise experimental evidence brief—not paper
prose—that maps each artifact to the planned Threat Model, Architecture,
Security Invariant, Evaluation, Results, and Limitations sections.
