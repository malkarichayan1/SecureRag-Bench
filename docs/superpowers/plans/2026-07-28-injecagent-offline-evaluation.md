# InjecAgent Offline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce stratified, reproducible offline InjecAgent payload-transfer results for all public base and enhanced case files.

**Architecture:** Extend normalized case metadata, group case-level outcomes in the offline runner, and add a dedicated CLI study mode. The runner retains a generic retrieval-to-simulated-email path and explicitly labels results as offline payload-transfer evaluation.

**Tech Stack:** Python 3.10+, pytest, standard-library JSON/pathlib, existing SecureRAG-Bench interpreter and red-team harness.

## Global Constraints

- Do not call a hosted model, paid API, or real external tool.
- Label every result `offline_payload_transfer`; do not claim official InjecAgent metrics.
- Preserve the existing single-file `injecagent` CLI behavior.
- Use test-first development and run the full suite before artifact handoff.

---

### Task 1: Preserve benchmark metadata

**Files:**
- Modify: `src/secure_rag_bench/evaluation/benchmark_adapters.py`
- Modify: `tests/test_benchmark_adapters.py`

**Interfaces:**
- Produces: `InjecAgentCase.attacker_tools: tuple[str, ...]`.
- Produces: `load_injecagent_cases(path, source_name=None) -> list[InjecAgentCase]` where each case retains its source name.

- [ ] **Step 1: Write the failing test**

```python
case = load_injecagent_cases(dataset_path, source_name="dh_base")[0]
assert case.source_name == "dh_base"
assert case.attacker_tools == ("GmailSendEmail",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\\Scripts\\python.exe -m pytest tests\\test_benchmark_adapters.py -q`

Expected: FAIL because `source_name` and `attacker_tools` are unavailable.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class InjecAgentCase:
    source_name: str
    attacker_tools: tuple[str, ...]

def load_injecagent_cases(path, source_name=None):
    source = Path(path)
    normalized_source = source_name or source.stem
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\\Scripts\\python.exe -m pytest tests\\test_benchmark_adapters.py -q`

Expected: PASS.

### Task 2: Add stratified offline results

**Files:**
- Modify: `src/secure_rag_bench/evaluation/offline_study.py`
- Modify: `tests/test_offline_study.py`

**Interfaces:**
- Produces: `run_injecagent_subset(...) -> dict[str, object]` with `evaluation_type`, `limitations`, `by_attack_type`, and `by_user_tool`.

- [ ] **Step 1: Write the failing test**

```python
result = run_injecagent_subset(dataset_path)
assert result["evaluation_type"] == "offline_payload_transfer"
assert result["by_attack_type"]["Data Stealing"]["case_count"] == 2
assert result["by_user_tool"]["GmailReadEmail"]["policy_halts"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\\Scripts\\python.exe -m pytest tests\\test_offline_study.py -q`

Expected: FAIL because stratified output keys are absent.

- [ ] **Step 3: Write minimal implementation**

```python
def _summarize_groups(rows, field):
    groups = {}
    for row in rows:
        group = groups.setdefault(row[field], {"case_count": 0, "attack_successes": 0, "policy_halts": 0})
        group["case_count"] += 1
        group["attack_successes"] += int(row["attack_succeeded"])
        group["policy_halts"] += int(row["policy_halted"])
    return groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\\Scripts\\python.exe -m pytest tests\\test_offline_study.py -q`

Expected: PASS.

### Task 3: Add the four-suite CLI study

**Files:**
- Modify: `src/secure_rag_bench/evaluation/run_eval.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Produces: CLI mode `injecagent-study --injecagent-dir PATH --output PATH`.
- Produces: JSON `injecagent_study` keyed by `dh_base`, `dh_enhanced`, `ds_base`, and `ds_enhanced`.

- [ ] **Step 1: Write the failing test**

```python
assert main(["injecagent-study", "--injecagent-dir", str(directory), "--output", str(output)]) == 0
assert set(result["injecagent_study"]) == {"dh_base", "dh_enhanced", "ds_base", "ds_enhanced"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\\Scripts\\python.exe -m pytest tests\\test_evaluation.py::test_cli_writes_injecagent_study_artifact -q`

Expected: FAIL because the mode is absent.

- [ ] **Step 3: Write minimal implementation**

```python
STANDARD_INJECAGENT_SUITES = {
    "dh_base": "test_cases_dh_base.json",
    "dh_enhanced": "test_cases_dh_enhanced.json",
    "ds_base": "test_cases_ds_base.json",
    "ds_enhanced": "test_cases_ds_enhanced.json",
}
```

Call `run_injecagent_subset` for each resolved file, require `--injecagent-dir`, and write the normal JSON output.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\\Scripts\\python.exe -m pytest tests\\test_evaluation.py::test_cli_writes_injecagent_study_artifact -q`

Expected: PASS.

### Task 4: Generate and document artifacts

**Files:**
- Create: `artifacts/injecagent_offline_study.json`
- Modify: `artifacts/README.md`

**Interfaces:**
- Consumes: `injecagent-study` CLI output.
- Produces: documented command and methodological limitation.

- [ ] **Step 1: Run the four-suite study**

Run:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval injecagent-study --injecagent-dir data\external\InjecAgent\data --output artifacts\injecagent_offline_study.json
```

- [ ] **Step 2: Document the artifact**

Add the exact command and state that the result maps source user tools to a generic retrieval operation and therefore is not an official InjecAgent score.

- [ ] **Step 3: Run final verification**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pip check
```

Expected: all tests pass and pip reports no broken requirements.
