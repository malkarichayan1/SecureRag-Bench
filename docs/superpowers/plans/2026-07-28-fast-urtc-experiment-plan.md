# Fast Defensible URTC Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the deterministic security evaluation into a transparent multi-configuration, multi-vector experiment package suitable for drafting.

**Architecture:** Extend only the existing red-team fixtures and ablation reporting. Each configuration uses the same tasks and attack plans; the reference monitor flags determine enforcement. A paper-evidence brief links results to claims without modifying the paper draft.

**Tech Stack:** Python 3.10+, pytest, existing CaMeL interpreter, deterministic mock planners/tools, JSON artifacts.

## Global Constraints

- No paid API, hosted model, local-model installation, or real external action.
- Do not claim an XML delimiter or deterministic planner is a prompt-defense baseline.
- Do not change `docs/paper_draft.md`.
- Use test-first implementation and run all tests before handoff.

---

### Task 1: Add multi-step benign and external-flow attack fixtures

**Files:**
- Modify: `src/secure_rag_bench/evaluation/red_team.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Produces: trusted workflow tasks and attack vectors `tainted_transfer_amount`, `tainted_post_body`.
- Produces: `_attack_plan(vector)` for both external-flow vectors.

- [ ] **Step 1: Write failing tests** asserting the task count increases and both
new attack paths are blocked by the full monitor.
- [ ] **Step 2: Run targeted tests** and observe failure because vectors are absent.
- [ ] **Step 3: Add minimal fixed fixture plans/tools** for the two paths.
- [ ] **Step 4: Run targeted tests** and observe success.

### Task 2: Add transparent configuration comparison

**Files:**
- Modify: `src/secure_rag_bench/evaluation/run_eval.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `run_monitor_ablation()` entries `no_monitor`, `xml_delimiters_only`, `policy_only`, and `full_monitor`.
- Produces: per-variant `policy_halt_rate` and per-vector outcomes.

- [ ] **Step 1: Write failing test** asserting the four configuration names and
the XML-only result equals no-monitor ASR under the controlled interpreter.
- [ ] **Step 2: Run targeted test** and observe failure because entries are absent.
- [ ] **Step 3: Add the minimal configuration map and halt-rate calculation.**
- [ ] **Step 4: Run targeted test** and observe success.

### Task 3: Generate reproducible experiment evidence

**Files:**
- Create: `artifacts/fast_urtc_controlled_evaluation.json`
- Create: `docs/experimental_evidence_brief.md`
- Modify: `artifacts/README.md`

**Interfaces:**
- Consumes: ablation, CEM aggregate, and InjecAgent study artifacts.
- Produces: a non-prose evidence map for paper sections and exact regeneration commands.

- [ ] **Step 1: Run the ablation CLI** and save its JSON output.
- [ ] **Step 2: Write evidence brief** with only tested claims and limitations.
- [ ] **Step 3: Run full pytest and pip check.**
