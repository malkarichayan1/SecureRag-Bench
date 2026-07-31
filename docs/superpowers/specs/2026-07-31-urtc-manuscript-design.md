# URTC SecureRAG-Bench Manuscript Design

## Purpose

Prepare a focused empirical paper for the IEEE MIT Undergraduate Research
Technology Conference (URTC) about SecureRAG-Bench. The paper answers how
retrieval-mediated prompt injections in RAG and tool-using LLM systems can be
mitigated without reducing legitimate-task utility.

## Working submission constraints

Until a current 2026 call for papers or template is available, use the supplied
URTC 2025 guidance as the working specification: English, at most five
single-spaced pages, at least 10-point text, an abstract below 500 words,
numbered tables and figures, and IEEE-style numbered references. The 2025
deadline and portal are not treated as current submission information.

## Contribution and claims

The paper is an empirical SecureRAG-Bench paper, not a general defense survey.
It presents a reproducible offline implementation of architectural isolation:
a privileged planner, a quarantined parser, restricted-AST execution, and a
provenance-aware reference monitor. The contribution is the implementation and
evaluation of this architecture, not the invention of CaMeL.

Paper-facing evidence is limited to:

- the 15-benign/7-attack controlled evaluation and four-configuration ablation;
- the ten-seed CEM retrieval-trigger experiment; and
- the 2,108-case offline InjecAgent payload-transfer study, explicitly not an
  official InjecAgent benchmark score.

The native Qwen 7B InjecAgent pilot is included only as a short, clearly
labeled preliminary paragraph. Its 50% valid-output rate is below the protocol
gate for a full native evaluation, so it is not a primary result or headline
benchmark claim.

## Manuscript structure and page budget

1. Title and abstract (about 0.35 page): state the threat, architecture, and
   headline controlled results.
2. Introduction and threat model (about 0.7 page): explain why optimized
   trigger fragments cross the retrieval boundary and why untrusted content
   must not choose capabilities.
3. System design (about 0.9 page): describe planner/parser separation,
   restricted interpreter, provenance labels, and dispatch policy.
4. Evaluation and results (about 1.6 pages): include one compact four-row
   ablation table, CEM trigger result, and offline payload-transfer result.
5. Preliminary native pilot, related work, and limitations (about 0.85 page):
   clearly distinguish model proposals, gated execution, provenance evidence,
   and offline replay.
6. Conclusion and references (about 0.6 page): concise conclusions and
   IEEE-style references.

If space permits, include one small system-architecture figure. It must
supplement, rather than duplicate, the text or table.

## Citation plan

Use citations supplied from NotebookLM through an exported bibliography,
reference list, or BibTeX file. Required anchors are CaMeL, AgentDojo,
InjecAgent, Task Shield, and the retrieval-barrier/CEM work. Additional work
on isolation, latent reasoning, defensive tokens, and runtime verification is
included only when it directly supports a concise related-work comparison.

## Quality and verification

The LaTeX source will compile to a PDF that stays within five pages. Before
delivery, render and visually inspect the PDF; verify table/figure labels,
citations, page count, and that the preliminary pilot is not presented as an
official native InjecAgent score. Confirm the current URTC requirements before
submission.
