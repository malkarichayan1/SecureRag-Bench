# Manuscript Revision Notes for the Native Evaluation

Apply these changes only after native artifacts have completed the protocol in
`docs/native_injecagent_protocol.md` and passed artifact review.

## Contribution framing

Describe SecureRAG-Bench as a reproducible implementation and evaluation of an
architectural prompt-injection defense, not as the invention of the CaMeL
architecture. Cite:

- E. Debenedetti et al., "Defeating Prompt Injections by Design," 2025,
  https://arxiv.org/abs/2503.18813.
- E. Debenedetti et al., "AgentDojo: A Dynamic Environment to Evaluate Prompt
  Injection Attacks and Defenses for LLM Agents," NeurIPS 2024.
- Q. Zhan et al., "InjecAgent: Benchmarking Indirect Prompt Injections in
  Tool-Integrated Large Language Model Agents," Findings of ACL 2024.
- F. Jia et al., "The Task Shield: Enforcing Task Alignment to Defend Against
  Indirect Prompt Injection in LLM Agents," ACL 2025.

## Native-model results section

Report two columns for each condition:

1. Model proposal ASR using the official InjecAgent score fields.
2. Execution ASR after the task-alignment gate.

State directly that the native task-alignment gate tests tool-choice blocking,
not provenance. Do not use a native tool-choice result as evidence that
provenance improves over policy-only enforcement.

## Provenance results section

Retain the controlled and offline authorized-action replay results as the
provenance-specific evidence. Their key comparison is policy-only versus full
monitor when an otherwise authorized action carries tool-derived data.

## Required limitations

- The primary native evaluation uses one pinned local model unless a second
  model has completed the same matrix.
- No live external tool call occurs.
- The native task-alignment gate is not a comparison of task alignment versus
  provenance.
- The offline replay maps public payloads into a generic simulated boundary and
  is not an official InjecAgent score.
- The restricted-AST guarantee applies only to supported operations and
  configured capabilities.
