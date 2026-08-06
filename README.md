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

Freeze a serialized 25 direct-harm/25 data-stealing held-out split, then run a
checkpointed pilot for the selected prompt condition:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.local_injecagent --setting base --model-config artifacts\native\model-qwen25-7b.json --prompt-condition strict_react --case-ids artifacts\native\splits\base-validity.json --checkpoint artifacts\native\checkpoints\qwen25-7b-base-pilot-strict.jsonl --only-first-step --output artifacts\native\runs\qwen25-7b-base-pilot-strict.json
.venv\Scripts\python.exe scripts\analyze_native_injecagent.py artifacts\native\runs\qwen25-7b-base-pilot-strict.json --output artifacts\native\analysis\qwen25-7b-base-pilot-gate.json
```

The output records the 90% validity decision and stable failure reasons. A full
qualifying run generates once; defenses are then replayed from the same
integrity-checked checkpoint without another model call:

```powershell
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.local_injecagent --setting base --model-config artifacts\native\model-qwen25-7b.json --prompt-condition strict_react --checkpoint artifacts\native\checkpoints\qwen25-7b-base-full-strict.jsonl --defense no_defense --output artifacts\native\runs\qwen25-7b-base-full-no-defense.json
.venv\Scripts\python.exe -m secure_rag_bench.evaluation.local_injecagent --setting base --prompt-condition strict_react --checkpoint artifacts\native\checkpoints\qwen25-7b-base-full-strict.jsonl --replay-defense task_alignment_guard --output artifacts\native\runs\qwen25-7b-base-full-task-alignment.json
```

Read `docs/native_injecagent_protocol.md` for the exact split, calibration,
base/enhanced, clean-control, optional-model, gate-inspection, and aggregation
commands. A native InjecAgent task-alignment result is not a provenance
ablation; use the authorized-action replay artifacts for the provenance claim.
Official model ASR, explanatory validity diagnostics, and offline execution ASR
are reported separately.

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

## Kaggle native validity and adaptive attack study

`notebooks/securerag_native_adaptive_kaggle.ipynb` runs the complete native
InjecAgent validity study and the restricted-AST/adaptive-attack evaluation
on a Kaggle GPU instance, then exports a bundle for the `paper/urtc/`
manuscript. Regenerate the notebook from its typed cell sources with
`python scripts/build_kaggle_notebook.py` after editing
`scripts/build_kaggle_notebook.py`; do not hand-edit the `.ipynb` file. As of
this writing no real Kaggle bundle has been produced or imported --
`paper/generated/` still contains only `.gitkeep`, and
`docs/submission_readiness.md` records the `paper/urtc/main.tex` manuscript
as **BLOCKED** until one is. The steps below describe the procedure a user
runs to unblock it.

1. **Upload/open the notebook on Kaggle.** Create a new Kaggle Notebook and
   either upload `notebooks/securerag_native_adaptive_kaggle.ipynb` directly
   or attach this repository as a Kaggle Dataset/Notebook and open it from
   there. The first code cell clones this study's branch of the repository
   (if `pyproject.toml` is not already present -- the study's code lives on
   `codex/native-validity-adaptive`, not the default branch) and installs
   the `local-injecagent` extra.
2. **Enable a GPU accelerator.** Under Kaggle's Notebook Settings, set
   Accelerator to a GPU before running the pilot or full-native stages; the
   `transformers`-backed catalog models (`qwen-7b`, `qwen-14b`,
   `llama-3.1-8b`, `qwen-32b-4bit` in `configs/native_study_models.json`) load
   onto the GPU and will not fit in a reasonable time on CPU alone.
3. **Hugging Face license and token setup.** `meta-llama/Llama-3.1-8B-Instruct`
   is a gated Hugging Face model: accept its license on the model page with
   the Hugging Face account you intend to use, then add that account's access
   token as a Kaggle Secret named `HF_TOKEN` and attach it to the notebook
   (Kaggle's Secrets panel -> Attach to notebook). Attaching alone does not
   create an environment variable -- Kaggle only makes an attached secret
   retrievable through `kaggle_secrets.UserSecretsClient`, so the preflight
   cell reads any attached `HF_TOKEN`/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`
   secret through that client and copies it into `os.environ` itself before
   reporting credential presence. Never type a token literally into a cell.
   `Qwen/Qwen2.5-*` catalog entries are not gated and do not require a token,
   but leaving `HF_TOKEN` set is harmless.
4. **Optional credentials.** The `llama-3.1-70b-endpoint` and `claude` catalog
   entries are `enabled: false` by default and only usable if attempted
   explicitly; they need `OPENAI_API_KEY` (OpenAI-compatible endpoint) or
   `ANTHROPIC_API_KEY` respectively, attached the same way as `HF_TOKEN` -- as
   a Kaggle Secret, never hardcoded. A model whose required credential is
   absent is recorded with `status: "skipped"` and a `missing_credential:*`
   reason and excluded from the run rather than causing a failure.
5. **Read the preflight cell's output.** Stage 1 (`preflight`) prints GPU
   availability/device name(s), free disk space, installed package versions,
   and a `credential_presence` block that reports only `true`/`false` for
   `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `HF_TOKEN` -- values are never
   printed. Confirm `gpu_available: true` and every credential you intend to
   use reports `true` before continuing.
6. **Review the pilot-stage gate.** Stage 2 (`pilot`) builds the deterministic
   25 direct-harm + 25 data-stealing held-out split and runs each attempted
   model's held-out pilot, then prints its protocol-valid rate and
   `gate.passed`. This is the **90% validity gate**: a model qualifies only if
   at least 90% of its held-out pilot cases are protocol-valid with no
   runner, traceability, or integrity failure
   (`evaluate_validity_gate` in
   `src/secure_rag_bench/evaluation/native_analysis.py`). A
   qualifying model proceeds to stage 3 automatically. A model that fails the
   gate stops stage 3 (`full_native`) with `RuntimeError("validity gate not
   passed")` before that model's full run starts -- it does not silently
   downgrade, substitute a different result, or continue past the failure.
   To proceed, either fix the model/prompt condition or remove that model
   from `SELECTED_MODEL_NAMES` (edited in the pilot-setup cell) and re-run.
7. **Run the full native study.** Stage 3 (`full_native`) generates against
   every case in the study setting's InjecAgent pool for each model that
   passed its pilot gate -- on the order of 1,000 direct-harm and
   data-stealing cases per qualifying model, a multi-hour run per model
   against a metered Kaggle GPU quota. For a first trial, restrict
   `SELECTED_MODEL_NAMES` to a single model before running this stage.
8. **Checkpoints and resumability.** Every model-generation cell (stages 2,
   3, and the AST plan-generation cell in stage 4) writes each completed case
   to a `JsonlCheckpointStore` under `OUTPUT_ROOT/checkpoints/`. If a Kaggle
   session times out or is interrupted, simply re-run the same cell: it will
   resume from the checkpoint and only generate the cases that are not yet
   recorded there, rather than regenerating everything from scratch. To
   preserve progress across sessions, download the `checkpoints/` directory
   (or the whole `OUTPUT_ROOT` working directory) from Kaggle's Output tab
   before a session ends.
9. **Export the study bundle.** Stage 5 (`export`) assembles `bundle.json`
   and a file manifest over everything the earlier stages wrote, then invokes
   `scripts/export_study_bundle.py` to validate the resulting directory
   (`validate_study_bundle`) and package it into a single reproducible zip.
   Download that archive from Kaggle's Output tab.
10. **Import it locally.** From a local checkout of this repository, run
    `python scripts/import_study_bundle.py <downloaded-bundle>.zip --output-dir
    paper/generated`. This re-validates the bundle and, only if validation
    passes, writes `paper/generated/{native_validity,adaptive_monitor,
    ast_compatibility}.tex/.csv`, `macros.tex`, two plot-figure `.tex`
    bodies, `summary.json`, and a fresh `MANIFEST.sha256` -- atomically, so a
    failed or interrupted import never leaves `paper/generated/` partially
    written.
11. **Build the manuscript.** Run `make -C paper/urtc pdf` to typeset
    `paper/urtc/main.tex` against the freshly generated inputs, then run
    `python scripts/verify_urtc_pdf.py output/pdf/securerag_bench_urtc_draft.pdf`
    and visually inspect every rendered page (clipped text, overlapping
    tables, broken references, unreadable plots, page overflow) before
    treating any number in it as final. Cross-check every quoted figure
    against `docs/claim_traceability.md`'s `paper/urtc/main.tex` table.
12. **Optional and skipped models are excluded from every empirical table.**
    `build_paper_tables` (`src/secure_rag_bench/evaluation/study_reporting.py`)
    only includes a bundle configuration whose `status` is `"completed"`; any
    model recorded as `"skipped"` (missing credential, insufficient VRAM) or
    `"failed"` (held-out gate not passed) contributes no row to
    `native_validity`, `adaptive_monitor`, or `ast_compatibility`, and no
    skipped/excluded model's non-result is ever presented as if it were a
    measured outcome.
