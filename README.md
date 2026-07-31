# SecureRAG-Bench

SecureRAG-Bench is an offline, deterministic research baseline for evaluating
architectural defenses against indirect prompt injection in RAG agents.

The reference design separates a privileged planner from a quarantined parser.
A restricted-AST CaMeL interpreter is the sole tool dispatcher and will enforce
provenance, capability, and contextual-policy rules.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

The default test suite is fully offline. OpenAI-backed adapters are optional and
are not required for the baseline.

## Baseline verification

Run the deterministic baseline before and after each milestone:

```powershell
python -m pytest tests/test_package_smoke.py tests/test_scaffold_fixtures.py -v
```

## Interpreter verification

```powershell
.venv\Scripts\python.exe -m pytest tests/test_provenance.py tests/test_security_policy.py tests/test_interpreter.py -v
```

## Retrieval and trigger verification

```powershell
.venv\Scripts\python.exe -m pytest tests/test_retrieval.py tests/test_cem_engine.py -v
```

## External benchmark handoff

The offline red-team command measures the deterministic CaMeL target only; it
does not represent a published AgentDojo or InjecAgent score. Use the adapters
to keep the external benchmark inputs and native runners separate.

### AgentDojo

Install its optional dependency, then use AgentDojo's native environment and
runner (replace the model and attack with the experiment configuration):

```powershell
.venv\Scripts\python.exe -m pip install -e ".[agentdojo]"
.venv\Scripts\python.exe -m agentdojo.scripts.benchmark --suite workspace --model gpt-4o-2024-05-13 --attack tool_knowledge --logdir runs/agentdojo
```

### InjecAgent

Clone the official dataset repository separately, then normalize either
`data/test_cases_dh_base.json` or `data/test_cases_ds_base.json` before mapping
its user tool to a simulated tool in this project:

```powershell
.venv\Scripts\python.exe -c "from secure_rag_bench.evaluation import load_injecagent_cases; print(len(load_injecagent_cases(r'InjecAgent/data/test_cases_dh_base.json')))"
```

External benchmark runs and an API-backed jury require their own credentials,
tool mappings, and recorded experiment configuration. They are intentionally
not invoked by the offline test suite.

### Native local-model InjecAgent protocol

The native prompted-agent runner uses InjecAgent's parser and score fields but
records the model's proposed attack separately from a simulated task-alignment
execution decision. Install the optional local stack on a CUDA-capable host:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[local-injecagent]"
```

Run a stratified base-setting pilot with 25 direct-harm and 25 data-stealing
cases for each condition:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.local_injecagent --model Qwen/Qwen2.5-7B-Instruct --setting base --max-cases-per-attack 25 --only-first-step --defense no_defense --output artifacts\native_injecagent_qwen25_7b_base_no_defense_pilot.json
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.local_injecagent --model Qwen/Qwen2.5-7B-Instruct --setting base --max-cases-per-attack 25 --only-first-step --defense task_alignment_guard --output artifacts\native_injecagent_qwen25_7b_base_task_alignment_guard_pilot.json
```

Then summarize conditions with:

```powershell
.venv\Scripts\python.exe scripts\analyze_native_injecagent.py artifacts\native_injecagent_qwen25_7b_base_no_defense_pilot.json artifacts\native_injecagent_qwen25_7b_base_task_alignment_guard_pilot.json --output artifacts\native_injecagent_qwen25_7b_base_pilot_summary.json
```

Read `docs/native_injecagent_protocol.md` before scaling this experiment. A
native InjecAgent tool-choice result is not a provenance ablation; use the
authorized-action replay artifacts for the provenance claim. For two-stage
data-stealing cases, use the no-defense artifact for complete model-vulnerability
scores and guarded artifacts for executed-action scores.

### Free offline baseline comparison

The local InjecAgent files can also be evaluated across six deterministic
variants without an API key or network connection:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval injecagent-baselines --injecagent-dir data\external\InjecAgent\data --output artifacts\injecagent_free_baselines.json
```

The variants are no defense, XML delimiters, a keyword sanitizer, a
task-alignment guard, policy-only enforcement, and the full monitor. The
sanitizer and alignment guard are deterministic surrogates inspired by defense
families, not reproductions of PromptArmor or Task Shield. This is not an
official InjecAgent score.
