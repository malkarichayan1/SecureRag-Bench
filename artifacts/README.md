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

## Adaptive attack/control scenario sweep

`adaptive_scenarios_sweep.json` runs the fixed 10-pair (20-scenario) adaptive
catalog -- CEM-integrated retrieval, multi-step provenance transformations,
split/obfuscated content, and policy-aware authorized actions, each with a
paired benign control -- under all three monitor configurations
(`no_monitor`, `policy_only`, `full_monitor`; 60 runs total). Regenerate it
with:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval cem --output artifacts\cem_for_adaptive.json
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval adaptive --cem-artifact artifacts\cem_for_adaptive.json --output artifacts\adaptive_scenarios_sweep.json
```

`--cem-artifact` must point at a JSON file supplying a `"best_prefix"`
string, in either of two shapes: a flat `{"best_prefix": "..."}` file (what
`run_cem_eval()` itself returns, e.g. if a caller saves that value
directly), or the shape `run_eval cem --output` actually writes on disk --
nested under a `"cem"` key, i.e. `{"cem": {"best_prefix": "...", ...}}` --
since `main()` always nests every mode's result under its mode name. The
command above uses the second (real, CLI-produced) shape. `cem_seed_11.json`
/ `cem_seed_22.json` / etc. in this directory come from `study` mode
instead, whose shape has no `best_prefix` key at all and is **not** a valid
`--cem-artifact` input; only a `cem`-mode artifact (or a hand-crafted flat
file) works here. Only the single `best_prefix` string is used, wrapped in a
one-element list.

The written artifact has three top-level keys under `"adaptive"`:
`"records"` (every one of the 60 runs, flattened to JSON-safe primitives --
see `AdaptiveRunRecord` in `adaptive_runner.py`), `"summary"`
(`summarize_adaptive_records`'s grouped rates -- overall, by monitor, by
family-then-monitor, by scenario-then-monitor, each with counts,
denominators, and Wilson 95% intervals), and `"manifest"` (a SHA-256-hashed
integrity envelope over every record, for drift detection).

**This is a deterministic-plan study, not a native model-generation study.**
Every scenario's `plan` is fixed, hand-authored Python text baked into
`adaptive_scenarios.py` -- it is fed straight to the real `CaMeLInterpreter`
and real retrieval pipeline, but no language model ever generates it. This
isolates the interpreter/policy/provenance enforcement boundary's behavior
in isolation: "if an attacker's payload already reached this exact plan
shape, would the monitor stop it." It is a completely separate question from
whether a real language model, prompted end-to-end, would actually produce
such a plan in the first place -- that question is answered by the native
validity pipeline (`native_analysis.py`, `native_validity.py`,
`local_injecagent.py`), which drives a real model adapter instead. Do not
report this suite's ASR figures as if they were native-model attack success
rates; report them as an upper bound on what the boundary can catch once a
given attack shape reaches it.

Every simulated tool target in the catalog is synthetic: emails go to
`team@example.test` (an RFC 2606 reserved test domain), and transfer
accounts/post channels use unambiguous placeholder tokens (e.g.
`VENDOR-ONFILE-002`). `run_adaptive_eval` calls
`adaptive_analysis.assert_no_real_targets` before writing any artifact,
which raises if a recorded call ever names an email at a domain outside the
reserved `example.com`/`example.net`/`example.org`/`*.test`/`*.example`/
`*.invalid`/`*.localhost` set -- see that function's docstring for exactly
what the heuristic does and does not cover.

## Benign restricted-AST compatibility

`ast_compatibility_<source>.json` reports how often raw, *unrepaired*
plan text is accepted by the restricted-AST grammar and executes to its
declared benign effect, across the five-family, 25-case catalog in
`ast_compatibility.py` (email, calendar, repository, banking,
external_posting). Regenerate it with:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.run_eval ast-compatibility --records artifacts\ast_compatibility_records.json --output artifacts\ast_compatibility_<source>.json
```

`--records` must point at a JSON file containing a non-empty list of
objects, each with:

- `"case_id"` (str): must match a `BenignPlanCase.case_id` from
  `build_benign_plan_catalog()` (e.g. `"email-001"`, `"banking-003"`; see
  `ast_compatibility.py` for the full id list).
- `"raw_plan"` (str): the raw plan text to feed **unmodified** into
  `evaluate_benign_plan` -- it is never repaired before validation.

Example:

```json
[
  {"case_id": "email-001", "raw_plan": "send_email('team@example.test', 'Status', 'On track')"},
  {"case_id": "banking-001", "raw_plan": "transfer('vendor-001', 250.0)"}
]
```

**Nothing else in this repository currently produces this file.** It is
meant to hold real model-generated plan text -- e.g. from a future
generation pass that prompts a model with each case's `user_query` and
records its raw completion -- or a hand-authored fixture for a specific
regression. Repeated `case_id` entries are allowed (e.g. several generations
of the same case across seeds); the written manifest keys each record by
`"<case_id>::<index>"` so duplicates still hash uniquely.

The written artifact has three top-level keys under `"ast_compatibility"`:
`"records"` (every `ASTCompatibilityRecord`), `"summary"`
(`summarize_ast_compatibility`'s totals, per-family breakdown, and rejection
taxonomy, each with Wilson 95% intervals), and `"manifest"` (a SHA-256-hashed
integrity envelope). Like the adaptive sweep above, this evaluates plans
against the interpreter directly -- it does not itself generate plans from a
model, so it measures grammar/effect compatibility for whatever plan text
`--records` supplies, not a model's overall plan-generation success rate.

## Kaggle study bundle (native validity, restricted-AST, adaptive attacks)

The artifacts above are all generated by this repository's own offline test
suite and mock/deterministic components. A separate, larger evidence package
-- native local-model InjecAgent validity across the full catalog, benign
restricted-AST compatibility, and the adaptive attack sweep, all against real
model generations -- is produced by
`notebooks/securerag_native_adaptive_kaggle.ipynb` on a Kaggle GPU instance
and is not stored in this directory. That notebook exports a validated
**study bundle** (`bundle.json`, `MANIFEST.sha256`, and the `splits/`,
`checkpoints/`, `runs/`, `ast/`, `adaptive/`, `environment/` trees documented
in `secure_rag_bench.evaluation.study_reporting`'s module docstring) with
`scripts/export_study_bundle.py`, which validates the bundle
(`validate_study_bundle`) before zipping it. Import a downloaded bundle
locally with:

```powershell
.venv\Scripts\python.exe scripts\import_study_bundle.py <bundle>.zip --output-dir paper\generated
```

This writes the `paper/urtc/main.tex` manuscript's generated tables, plot
data, and headline macros -- see `README.md`'s "Kaggle native validity and
adaptive attack study" section for the full user workflow and
`docs/claim_traceability.md`'s `paper/urtc/main.tex` table for exactly which
generated file backs which manuscript claim. As of this writing no real
bundle has been imported (`paper/generated/` contains only `.gitkeep`); see
`docs/submission_readiness.md` for that manuscript's current BLOCKED status.

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
