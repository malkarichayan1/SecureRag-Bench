# Offline Result Artifacts

All files in this directory are generated with deterministic mock components;
no API key, network model call, or real external action is used.

## Reference-monitor ablation

Regenerate `monitor_ablation.json` with:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval ablation --output artifacts\monitor_ablation.json
```

The current expanded artifact, `expanded_offline_ablations.json`, reports 12
benign tasks and five attacks across three configurations: the full monitor,
provenance checks disabled, and the monitor disabled.

For the final fast-path controlled comparison, regenerate
`fast_urtc_controlled_evaluation.json` with:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval ablation --output artifacts\fast_urtc_controlled_evaluation.json
```

It reports 15 benign tasks and seven attack paths across no monitoring, XML
delimiters without enforcement, policy-only enforcement, and the full monitor.
It is a deterministic controlled evaluation; it is not a real-model
prompt-following comparison.

## InjecAgent payload subset

`injecagent_dh_base_25.json` records a bounded run over 25 public InjecAgent
payloads using the offline target and simulated tools:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval injecagent --injecagent-path data\external\InjecAgent\data\test_cases_dh_base.json --limit 25 --output artifacts\injecagent_dh_base_25.json
```

This is a reproducible payload-transfer check, **not** an official InjecAgent
benchmark score: the adapter maps each source tool to a generic retrieval
operation and evaluates whether untrusted retrieved text can reach a simulated
external email action. The paper should report it with that limitation.

## Full InjecAgent offline study

`injecagent_offline_study.json` is the complete free/offline payload-transfer
study over the four public base/enhanced files (2,108 cases total):

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval injecagent-study --injecagent-dir data\external\InjecAgent\data --output artifacts\injecagent_offline_study.json
```

It contains case-level outcomes and ASR/policy-halt summaries stratified by
attack type and source user tool. As with the bounded subset above, this maps
each benchmark user tool to generic untrusted retrieval and a simulated email
action. It is therefore an **offline payload-transfer study**, not an official
InjecAgent ASR-valid or ASR-all result.

## Free InjecAgent baseline comparison

`injecagent_free_baselines.json` compares six fully offline variants over the
same four public source files:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval injecagent-baselines --injecagent-dir data\external\InjecAgent\data --output artifacts\injecagent_free_baselines.json
```

The no-defense, XML-delimiters-only, keyword-sanitizer, task-alignment-guard,
policy-only, and full-monitor variants share a generic retrieval and simulated
email boundary. The keyword sanitizer and task-alignment guard are deterministic
surrogates inspired by defense families. They are not PromptArmor or Task
Shield reproductions, and this artifact is not an official InjecAgent score.

## Native local-model InjecAgent evaluation

The native runner in `secure_rag_bench.evaluation.local_injecagent` executes
the official prompted-agent parser with a local model. It stores model-level
native scores separately from simulated task-alignment execution scores. The
latter measures whether a proposed attacker tool would execute; it is not a
provenance ablation because native InjecAgent treats attacker-tool selection as
the successful event.

See `docs/native_injecagent_protocol.md` for the fixed protocol. Artifacts
from a completed run should use the names
`native_injecagent_<model>_<setting>_<defense>.json`; summarize matched
conditions with:

```powershell
.venv\Scripts\python.exe scripts\analyze_native_injecagent.py artifacts\native_injecagent_qwen25_7b_base_no_defense.json artifacts\native_injecagent_qwen25_7b_base_task_alignment_guard.json --output artifacts\native_injecagent_qwen25_7b_base_summary.json
```

## CEM trigger study

Regenerate each full 30-iteration, 5,000-sample seed independently:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval study --seeds 11 --output artifacts\cem_seed_11.json
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval study --seeds 22 --output artifacts\cem_seed_22.json
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval study --seeds 33 --output artifacts\cem_seed_33.json
```

`cem_study_aggregate.json` merges those three saved study payloads. It contains
the mean and population standard deviation of best cosine fitness and the
fraction of final triggers that place the malicious document in the top five.
Regenerate it with:

```powershell
.venv\Scripts\python.exe scripts\aggregate_cem_study.py artifacts\cem_seed_11.json artifacts\cem_seed_22.json artifacts\cem_seed_33.json --output artifacts\cem_study_aggregate.json
```

`cem_extended_10.json` is the stronger paper-facing robustness run: ten
independent full-protocol seeds (30 iterations and 5,000 candidate prefixes per
iteration). Regenerate it with:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval study --seeds 1 2 3 4 5 6 7 8 9 10 --output artifacts\cem_extended_10.json
```

## Verification

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pip check
```

`MANIFEST.sha256` records the SHA-256 digests of the three paper-facing result
artifacts. Regenerate it after an intentional experiment re-run; otherwise use
it to detect accidental artifact drift during manuscript preparation.

## Local review PDF

Regenerate the non-template review PDF from the current local draft with:

```powershell
.venv\Scripts\python.exe scripts\generate_paper_pdf.py
```

The output is `output\pdf\securerag_bench_offline_draft.pdf`. It is a local
review artifact, not a URTC-formatted submission PDF.
