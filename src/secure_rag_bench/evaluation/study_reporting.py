"""Cross-artifact validation and paper-ready aggregation for a study bundle.

A *study bundle* is the single, self-contained directory a Kaggle run exports
and a local machine imports. It is the only input from which manuscript
numbers may be generated: ``validate_study_bundle`` is the gate, and
``build_paper_tables`` / ``build_plot_series`` refuse to invent anything the
gate did not confirm.

Nothing in this repository produced such a directory before this module; the
schema below is normative and is what ``scripts/export_study_bundle.py`` must
write and ``scripts/import_study_bundle.py`` must read.

## Where the code lives

This module is the **only** import site callers need. It re-exports every
public name, so ``from secure_rag_bench.evaluation.study_reporting import ...``
is the supported entry point and the two modules below it are internal::

    study_reporting             <- import from here
      the normative schema (this docstring), ``validate_study_bundle``,
      ``ValidatedStudyBundle``, ``PaperTables`` and its row dataclasses,
      ``build_paper_tables``, ``build_plot_series``
        |
        v
    study_bundle_validation     internal
      artifact-specific validation chains (native stages, AST runs, adaptive
      runs, splits, environment captures)
        |
        +---------------------------+
        |                           |
        v                           v
    study_bundle_io             study_bundle_model
      generic read-only             the frozen dataclasses a validated
      primitives: file              bundle is made of (BundleModel,
      manifests, JSONL              SplitManifest, NativeStage,
      envelopes, digests,           NativeConfiguration, ASTRun,
      secret scanning, JSON         AdaptiveRun, NativeReplayArtifact)
      shape assertions

The two bottom modules are leaves: ``study_bundle_model`` imports nothing from
this package at all. The dependency arrows are strictly one-way; adding an
import in the other direction would create a cycle.

## Directory schema (``schema_version`` 1)

```
<root>/
  bundle.json                          # REQUIRED index; the only entry point
  MANIFEST.sha256                      # REQUIRED file-integrity manifest
  splits/<split_id>.json               # REQUIRED (>= 1) held-out split manifest
  checkpoints/<name>.jsonl             # raw native generation JSONL, one per stage
  runs/<name>.json                     # native run output (scores + gate_decision)
  runs/<name>.manifest.json            # StudyManifest envelope for that stage
  replay/<name>.json                   # offline defense replay of a stage
  ast/<run_id>.json                    # {"records","summary","manifest"}
  adaptive/<run_id>.json               # {"records","summary","manifest"}
  environment/<name>.json              # GPU/package/timestamp capture
```

Every directory other than ``splits/`` is optional *as a directory*: it must
exist exactly when ``bundle.json`` references a file inside it. There are no
other files; ``MANIFEST.sha256`` must list every file in the tree (itself
excluded) and nothing else, so a stray lock file, ``.corrupt`` quarantine, or
editor backup is a validation failure rather than silently ignored content.

### File naming is declared, never parsed

Two pre-existing documents in this repository disagree about native artifact
names (``docs/native_injecagent_protocol.md`` uses
``qwen25-7b-base-pilot-strict.json``; ``artifacts/README.md`` claims
``native_injecagent_<model>_<setting>_<defense>.json``). Neither is adopted as
normative here. **Validation never infers meaning from a filename** -- every
path is declared explicitly in ``bundle.json``, so the ambiguity cannot leak
into behaviour. For human readability an exporter *should* use:

- ``<configuration_id> = <model>-<setting>-<prompt_condition>``
- ``checkpoints/<configuration_id>-{pilot,full}.jsonl``
- ``runs/<configuration_id>-{pilot,full}.json`` and ``...manifest.json``
- ``replay/<configuration_id>-<stage>-<defense>.json``
- ``environment/<configuration_id>.json``

but any relative path under ``<root>`` is accepted. Declared paths must be
POSIX-style, relative, free of ``..`` segments, and resolve to a regular file.

**The sole exception is ``splits/<split_id>.json``.** ``bundle.json`` has no
``splits`` key: split manifests are discovered by globbing ``splits/*.json``,
and each file's *stem* is its ``split_id`` -- the identifier a configuration's
``split`` field must match. For splits, and only splits, the filename is
semantically load-bearing, so an exporter may not rename a split file without
also updating every ``configurations[].split`` that refers to it.

## ``bundle.json``

```jsonc
{
  "schema_version": 1,
  "study_id": "native-validity-adaptive",
  "created_utc": "2026-08-05T12:00:00Z",   // ISO-8601 UTC, trailing "Z"
  "models": [ ... ],                        // copy of configs/native_study_models.json's "models"
  "configurations": [ ... ],
  "ast_runs": [ ... ],
  "adaptive_runs": [ ... ]
}
```

``models`` is embedded rather than referenced so the bundle stays portable
once it leaves the machine that holds ``configs/native_study_models.json``.
Each entry needs ``name``, ``provider``, ``model_id``, ``tier`` and
``enabled``; names must be unique.

### ``configurations[]``

```jsonc
{
  "configuration_id": "qwen-7b-base-strict_react",  // unique within the bundle
  "model": "qwen-7b",                    // must name a models[] entry
  "tier": "primary",                     // must equal that entry's tier
  "setting": "base",                     // "base" | "enhanced"
  "prompt_condition": "strict_react",
  "status": "completed",                 // "completed" | "skipped" | "failed"
  "status_reason": null,                 // REQUIRED non-empty unless completed
  "split": "base-validity",              // splits/<split>.json
  "environment": "environment/....json", // REQUIRED when completed, else optional
                                         // (if present it is fully validated)
  "pilot": {                             // REQUIRED when completed
    "checkpoint": "checkpoints/....jsonl",
    "run": "runs/....json",
    "manifest": "runs/....manifest.json",
    "replays": []
  },
  "full": {                              // REQUIRED when completed, FORBIDDEN otherwise
    "checkpoint": "...", "run": "...", "manifest": "...",
    "replays": ["replay/....json"]
  }
}
```

``status`` is the field the plan's "only ``status == 'completed'``
configurations may be emitted" rule keys off. A ``skipped`` configuration
(optional model without credentials, conditional model without VRAM) may omit
every stage; a ``failed`` one may keep its pilot as evidence but must not
carry a ``full`` stage. **A ``completed`` configuration whose recomputed pilot
gate did not pass is rejected**, which is how "the full native stage must
refuse configurations below 90% held-out protocol validity" is enforced after
the fact rather than merely promised by the runner.

### ``ast_runs[]`` / ``adaptive_runs[]``

```jsonc
{"run_id": "ast-qwen-7b", "schema_version": 1, "model": "qwen-7b",
 "status": "completed", "status_reason": null, "path": "ast/ast-qwen-7b.json"}
```

``model`` is required for an AST run (the model whose raw plans were fed to
the interpreter) and absent for an adaptive run, which uses no language model.
Both files are exactly what ``run_eval.run_ast_compatibility_eval`` /
``run_eval.run_adaptive_eval`` return: ``{"records", "summary", "manifest"}``.

**Known gap:** neither ``ast_compatibility`` nor ``adaptive_analysis`` emits a
schema version of its own today. Rather than invent one inside those modules,
the *bundle entry* carries ``schema_version``, and this validator enforces the
exact record field set that version 1 means (listed in
``study_bundle_validation._AST_RECORD_FIELDS`` /
``study_bundle_validation._ADAPTIVE_RECORD_FIELDS``). If those record
dataclasses gain or lose a field, this constant must be bumped in lockstep;
both dataclass definitions carry a warning pointing back here.

## ``splits/<split_id>.json``

```jsonc
{"schema_version": 1, "setting": "base", "seed": 20260801,
 "calibration": ["<case_id>", ...], "held_out": ["<case_id>", ...]}
```

Richer than the ``{"calibration": [...], "held_out": [...]}` shape
``docs/native_injecagent_protocol.md`` writes by hand, because a bundle must
record *which* seed and setting produced the split. ``held_out`` must hold
exactly 50 unique ids disjoint from ``calibration``.

## ``environment/<name>.json``

No code in this repository captured environment metadata before this module,
despite the study design requiring "package versions, GPU metadata,
timestamps". This is the contract the Kaggle notebook must satisfy; only the
listed keys are required, extra keys are allowed and preserved.

```jsonc
{
  "schema_version": 1,
  "captured_utc": "2026-08-05T11:22:33Z",
  "platform": {"python_version": "3.11.9", "system": "Linux", "release": "6.1.85+"},
  "gpu": {"available": true,
          "devices": [{"name": "Tesla T4", "total_memory_gb": 15.0,
                       "driver_version": "550.90.07", "cuda_version": "12.4"}]},
  "packages": {"torch": "2.4.0", "transformers": "4.44.2", "secure-rag-bench": "0.1.0"}
}
```

``packages`` must additionally contain ``torch`` and ``transformers`` when the
configuration's catalog provider is ``transformers``; a hosted-endpoint
configuration has no local inference stack to pin.

## ``MANIFEST.sha256``

``sha256sum`` format -- ``<64 hex><two spaces><relative/posix/path>``, one line
per file, newline-terminated. Digests are over raw file bytes.

## What validation actually checks

1. **File hashes.** Every ``MANIFEST.sha256`` entry exists and matches; every
   file on disk is listed.
2. **Record hashes.** Every JSONL envelope's ``sha256`` recomputes, every
   record's own ``record_sha256`` recomputes, every ``run_identity_sha256``
   recomputes (via ``validated_run_identity``), and every ``StudyManifest``
   envelope recomputes from the records it claims to cover.
3. **Unique ids.** ``configuration_id``, ``run_id``, split id, and ``case_id``
   within a stage are unique; a run's records are a bijection onto its stage.
4. **Split membership.** A pilot stage covers its split's ``held_out`` exactly;
   a full stage covers ``held_out`` and ``calibration`` and everything its own
   manifest declares expected.
5. **25/25 pilot balance.** Exactly 25 ``dh`` and 25 ``ds`` held-out cases.
6. **Gate arithmetic.** ``evaluate_validity_gate`` is recomputed over the raw
   records and must equal the ``gate_decision`` the run file claims -- a run
   cannot assert a pass it did not earn.
7. **Full-run coverage.** No expected case id is missing from a stage.
8. **Model/dataset revision presence.** Every stage's shared run identity has a
   non-empty ``model.model_id`` matching the catalog, a non-empty
   ``model.model_revision``, and a ``dataset_revision`` that is neither empty
   nor ``"unknown"``. Hosted-endpoint configurations must pin a dated provider
   snapshot in ``model_revision``; "the API had no revision" is not a bundle
   this validator accepts.
9. **Raw-to-summary denominators.** ``scores``/``execution_scores`` are
   recomputed with ``compute_native_scores``/``compute_execution_scores``; AST
   and adaptive ``summary`` blocks are recomputed with their own summarizers.
   No summary is ever trusted on its own.
10. **Replay source hashes.** Every run/replay record must equal its checkpoint
    record plus *only* ``execution_step_1``/``execution_step_2``, and its
    ``record_sha256`` must still recompute after those keys are stripped, so a
    stale or edited replay is detectable.
11. **AST/adaptive schema versions.** As described above.
12. **Secret absence.** Any credential-shaped key whose value is not
    ``"[REDACTED]"`` is rejected, reusing ``redact_secrets``' own key matching
    so the two can never drift. A narrow token-shaped *value* scan
    (``sk-``/``hf_``/``ghp_``/``Bearer ...``) also runs, except inside the
    verbatim benchmark/model-output fields in
    ``study_bundle_io._VERBATIM_TEXT_KEYS``: that
    text is attacker-authored and must survive byte-for-byte for the official
    parser, so the key-based rule is the load-bearing guarantee there. This is
    a defence-in-depth backstop, not a substitute for redacting at write time.

Validation is strictly read-only. It deliberately re-implements the JSONL
envelope check instead of calling ``JsonlCheckpointStore.load_validated()``,
because that method takes a sibling ``.lock`` file and may quarantine and
rewrite a damaged trailing line -- both of which would mutate a bundle whose
integrity is the thing under test. The integrity primitive itself
(``record_digest``) is reused, and the envelope contract mirrored exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Mapping

from secure_rag_bench.evaluation.native_analysis import summarize_validity, wilson_interval
from secure_rag_bench.evaluation.study_bundle_io import (
    BUNDLE_INDEX_NAME,
    FILE_MANIFEST_NAME,
    REDACTION_SENTINEL,
    BundleValidationError,
    load_json,
    require_int,
    require_list,
    require_text,
    require_timestamp,
    require_unique,
    validate_file_manifest,
)
from secure_rag_bench.evaluation.study_bundle_model import (
    AdaptiveRun,
    ASTRun,
    BundleModel,
    NativeConfiguration,
    NativeReplayArtifact,
    NativeStage,
    SplitManifest,
)
from secure_rag_bench.evaluation.study_bundle_validation import (
    ADAPTIVE_SCHEMA_VERSION,
    AST_SCHEMA_VERSION,
    BUNDLE_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    HELD_OUT_PER_ATTACK,
    HELD_OUT_SIZE,
    SPLIT_SCHEMA_VERSION,
    load_splits,
    parse_models,
    validate_adaptive_run,
    validate_ast_run,
    validate_configuration,
)

__all__ = [
    "ADAPTIVE_SCHEMA_VERSION",
    "AST_SCHEMA_VERSION",
    "BUNDLE_INDEX_NAME",
    "BUNDLE_SCHEMA_VERSION",
    "ENVIRONMENT_SCHEMA_VERSION",
    "FILE_MANIFEST_NAME",
    "HELD_OUT_PER_ATTACK",
    "HELD_OUT_SIZE",
    "REDACTION_SENTINEL",
    "SPLIT_SCHEMA_VERSION",
    "AdaptiveMonitorRow",
    "AdaptiveRun",
    "ASTCompatibilityRow",
    "ASTRun",
    "BundleModel",
    "BundleValidationError",
    "ExecutionAsrRow",
    "NativeConfiguration",
    "NativeReplayArtifact",
    "NativeStage",
    "NativeValidityRow",
    "PaperTables",
    "SplitManifest",
    "ValidatedStudyBundle",
    "build_paper_tables",
    "build_plot_series",
    "validate_study_bundle",
]


# ---------------------------------------------------------------------------
# Validated bundle model
#
# The component dataclasses (``BundleModel``, ``SplitManifest``,
# ``NativeStage``, ``NativeConfiguration``, ``ASTRun``, ``AdaptiveRun``) are
# defined in ``study_bundle_validation`` next to the code that builds them and
# re-exported above; only the aggregate lives here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedStudyBundle:
    """Everything a validated bundle contains, parsed once and trusted after.

    This is the only input ``build_paper_tables`` and ``build_plot_series``
    accept: neither re-reads the filesystem, so a table can never describe a
    file that did not pass validation.
    """

    root: Path
    schema_version: int
    study_id: str
    created_utc: str
    models: tuple[BundleModel, ...]
    splits: tuple[SplitManifest, ...]
    configurations: tuple[NativeConfiguration, ...]
    ast_runs: tuple[ASTRun, ...]
    adaptive_runs: tuple[AdaptiveRun, ...]
    file_digests: Mapping[str, str]

    def model(self, name: str) -> BundleModel:
        """Return the catalog entry named ``name``."""
        for entry in self.models:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def split(self, split_id: str) -> SplitManifest:
        """Return the split manifest identified by ``split_id``."""
        for entry in self.splits:
            if entry.split_id == split_id:
                return entry
        raise KeyError(split_id)


# ---------------------------------------------------------------------------
# Paper tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionAsrRow:
    """Execution ASR for one defense applied to one configuration's full run.

    Nested inside ``NativeValidityRow`` rather than flattened into its own
    table row because validity is a property of the generation while execution
    ASR is a property of ``(generation, defense)``; a flat row per defense
    would duplicate every validity column.
    """

    defense: str
    count: int
    denominator: int
    rate: float
    wilson_95: dict[str, float]


@dataclass(frozen=True)
class NativeValidityRow:
    """One completed configuration's held-out gate plus full-study outcomes."""

    configuration_id: str
    model: str
    tier: str
    model_id: str
    setting: str
    prompt_condition: str
    held_out_denominator: int
    held_out_protocol_valid_count: int
    held_out_protocol_valid_rate: float
    held_out_protocol_valid_wilson_95: dict[str, float]
    gate_passed: bool
    case_count: int
    syntax_valid_count: int
    syntax_valid_rate: float
    syntax_valid_wilson_95: dict[str, float]
    protocol_valid_count: int
    protocol_valid_rate: float
    protocol_valid_wilson_95: dict[str, float]
    official_asr_valid_count: int
    official_asr_valid_denominator: int
    official_asr_valid_rate: float | None
    official_asr_valid_wilson_95: dict[str, float] | None
    official_asr_all_count: int
    official_asr_all_rate: float
    official_asr_all_wilson_95: dict[str, float]
    execution_asr: tuple[ExecutionAsrRow, ...]


@dataclass(frozen=True)
class AdaptiveMonitorRow:
    """One ``(family, monitor)`` cell of the adaptive attack table."""

    run_id: str
    family: str
    monitor: str
    case_count: int
    attack_count: int
    control_count: int
    retrieval_exposure_count: int
    retrieval_exposure_rate: float | None
    retrieval_exposure_wilson_95: dict[str, float] | None
    attempted_target_action_count: int
    attempted_target_action_rate: float | None
    attempted_target_action_wilson_95: dict[str, float] | None
    halt_count: int
    halt_rate: float | None
    halt_rate_wilson_95: dict[str, float] | None
    target_effect_asr_count: int
    target_effect_asr_denominator: int
    target_effect_asr: float | None
    target_effect_asr_wilson_95: dict[str, float] | None
    benign_utility_count: int
    benign_utility_denominator: int
    benign_utility: float | None
    benign_utility_wilson_95: dict[str, float] | None


@dataclass(frozen=True)
class ASTCompatibilityRow:
    """One ``(model, family)`` cell of the restricted-AST compatibility table."""

    run_id: str
    model: str
    family: str
    case_count: int
    accepted_count: int
    acceptance_rate: float
    acceptance_wilson_95: dict[str, float]
    execution_count: int
    execution_rate: float
    execution_wilson_95: dict[str, float]
    rejection_categories: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PaperTables:
    """The three primary manuscript tables, in deterministic row order."""

    native_validity: tuple[NativeValidityRow, ...]
    adaptive_monitor: tuple[AdaptiveMonitorRow, ...]
    ast_compatibility: tuple[ASTCompatibilityRow, ...]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def validate_study_bundle(root: str | Path) -> ValidatedStudyBundle:
    """Validate a complete study bundle and return its parsed contents.

    Raises ``BundleValidationError`` on the first failure, with a message that
    names the offending bundle-relative path. See the module docstring for the
    exact schema and the full list of checks.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise BundleValidationError(f"{root_path}: bundle root is not a directory")

    file_digests = validate_file_manifest(root_path)
    index = load_json(root_path, BUNDLE_INDEX_NAME)
    require_int(index, "schema_version", BUNDLE_INDEX_NAME)
    if index["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise BundleValidationError(
            f"{BUNDLE_INDEX_NAME}: unsupported schema_version "
            f"{index['schema_version']!r}; expected {BUNDLE_SCHEMA_VERSION}"
        )
    study_id = require_text(index, "study_id", BUNDLE_INDEX_NAME)
    created_utc = require_timestamp(index, "created_utc", BUNDLE_INDEX_NAME)

    models = parse_models(index)
    splits = load_splits(root_path)
    split_ids = {split.split_id for split in splits}

    configurations = tuple(
        validate_configuration(root_path, entry, models, splits, index_position)
        for index_position, entry in enumerate(require_list(index, "configurations", BUNDLE_INDEX_NAME))
    )
    require_unique(
        (configuration.configuration_id for configuration in configurations),
        label="configuration_id",
    )
    for configuration in configurations:
        if configuration.split_id not in split_ids:
            raise BundleValidationError(
                f"{BUNDLE_INDEX_NAME}: configuration {configuration.configuration_id!r} "
                f"names unknown split {configuration.split_id!r}"
            )

    ast_runs = tuple(
        validate_ast_run(root_path, entry, models)
        for entry in require_list(index, "ast_runs", BUNDLE_INDEX_NAME)
    )
    adaptive_runs = tuple(
        validate_adaptive_run(root_path, entry)
        for entry in require_list(index, "adaptive_runs", BUNDLE_INDEX_NAME)
    )
    require_unique((run.run_id for run in ast_runs), label="ast run_id")
    require_unique((run.run_id for run in adaptive_runs), label="adaptive run_id")

    return ValidatedStudyBundle(
        root=root_path,
        schema_version=BUNDLE_SCHEMA_VERSION,
        study_id=study_id,
        created_utc=created_utc,
        models=models,
        splits=splits,
        configurations=configurations,
        ast_runs=ast_runs,
        adaptive_runs=adaptive_runs,
        file_digests=dict(file_digests),
    )


# ---------------------------------------------------------------------------
# Paper tables and plot series
# ---------------------------------------------------------------------------


def build_paper_tables(bundle: ValidatedStudyBundle) -> PaperTables:
    """Build the three manuscript tables from a validated bundle.

    Only ``status == "completed"`` configurations and runs contribute rows, so
    a skipped optional model, a conditional model without VRAM, or a
    configuration that failed the held-out gate is structurally incapable of
    appearing as an empirical result.
    """
    return PaperTables(
        native_validity=tuple(
            _native_validity_row(configuration)
            for configuration in sorted(
                bundle.configurations, key=lambda item: item.configuration_id
            )
            if configuration.status == "completed"
        ),
        adaptive_monitor=tuple(
            row
            for run in sorted(bundle.adaptive_runs, key=lambda item: item.run_id)
            if run.status == "completed"
            for row in _adaptive_rows(run)
        ),
        ast_compatibility=tuple(
            row
            for run in sorted(bundle.ast_runs, key=lambda item: item.run_id)
            if run.status == "completed"
            for row in _ast_rows(run)
        ),
    )


def _native_validity_row(configuration: NativeConfiguration) -> NativeValidityRow:
    assert configuration.pilot is not None and configuration.full is not None
    gate = configuration.pilot.gate
    overall = summarize_validity([dict(record) for record in configuration.full.records])["overall"]
    syntax = overall["syntax_valid"]
    protocol = overall["protocol_valid"]
    asr_valid = overall["official_asr_valid"]
    asr_all = overall["official_asr_all"]

    execution = [
        _execution_row(configuration.full.run_defense, configuration.full.run_records)
    ]
    execution.extend(
        _execution_row(replay.defense, replay.records) for replay in configuration.full.replays
    )

    return NativeValidityRow(
        configuration_id=configuration.configuration_id,
        model=configuration.model,
        tier=configuration.tier,
        model_id=configuration.model_id,
        setting=configuration.setting,
        prompt_condition=configuration.prompt_condition,
        held_out_denominator=gate.protocol_valid_denominator,
        held_out_protocol_valid_count=gate.protocol_valid_count,
        held_out_protocol_valid_rate=gate.protocol_valid_rate,
        held_out_protocol_valid_wilson_95=dict(gate.wilson_95),
        gate_passed=gate.passed,
        case_count=overall["count"],
        syntax_valid_count=syntax["count"],
        syntax_valid_rate=syntax["rate"],
        syntax_valid_wilson_95=syntax["wilson_95"],
        protocol_valid_count=protocol["count"],
        protocol_valid_rate=protocol["rate"],
        protocol_valid_wilson_95=protocol["wilson_95"],
        official_asr_valid_count=asr_valid["count"],
        official_asr_valid_denominator=asr_valid["denominator"],
        official_asr_valid_rate=asr_valid["rate"],
        official_asr_valid_wilson_95=asr_valid["wilson_95"],
        official_asr_all_count=asr_all["count"],
        official_asr_all_rate=asr_all["rate"],
        official_asr_all_wilson_95=asr_all["wilson_95"],
        execution_asr=tuple(execution),
    )


def _execution_row(defense: str, records: Sequence[Mapping[str, Any]]) -> ExecutionAsrRow:
    execution = summarize_validity([dict(record) for record in records])["overall"][
        "execution_asr_all"
    ]
    return ExecutionAsrRow(
        defense=defense,
        count=execution["count"],
        denominator=execution["denominator"],
        rate=execution["rate"],
        wilson_95=execution["wilson_95"],
    )


def _adaptive_rows(run: AdaptiveRun) -> list[AdaptiveMonitorRow]:
    rows: list[AdaptiveMonitorRow] = []
    by_family = run.summary.get("by_family", {})
    for family in sorted(by_family):
        for monitor in sorted(by_family[family]):
            group = by_family[family][monitor]
            rows.append(
                AdaptiveMonitorRow(
                    run_id=run.run_id,
                    family=family,
                    monitor=monitor,
                    case_count=group["count"],
                    attack_count=group["attack_count"],
                    control_count=group["control_count"],
                    retrieval_exposure_count=group["retrieval_exposure_count"],
                    retrieval_exposure_rate=group["retrieval_exposure_rate"],
                    retrieval_exposure_wilson_95=group["retrieval_exposure_wilson_95"],
                    attempted_target_action_count=group["attempted_target_action_count"],
                    attempted_target_action_rate=group["attempted_target_action_rate"],
                    attempted_target_action_wilson_95=group["attempted_target_action_wilson_95"],
                    halt_count=group["halt_count"],
                    halt_rate=group["halt_rate"],
                    halt_rate_wilson_95=group["halt_rate_wilson_95"],
                    target_effect_asr_count=group["target_effect_asr_count"],
                    target_effect_asr_denominator=group["target_effect_asr_denominator"],
                    target_effect_asr=group["target_effect_asr"],
                    target_effect_asr_wilson_95=group["target_effect_asr_wilson_95"],
                    benign_utility_count=group["benign_utility_count"],
                    benign_utility_denominator=group["benign_utility_denominator"],
                    benign_utility=group["benign_utility"],
                    benign_utility_wilson_95=group["benign_utility_wilson_95"],
                )
            )
    return rows


def _ast_rows(run: ASTRun) -> list[ASTCompatibilityRow]:
    rejections: dict[str, dict[str, int]] = {}
    for record in run.records:
        category = record.get("rejection_category")
        if category:
            family = rejections.setdefault(str(record["family"]), {})
            family[str(category)] = family.get(str(category), 0) + 1

    rows: list[ASTCompatibilityRow] = []
    by_family = run.summary.get("by_family", {})
    for family in sorted(by_family):
        group = by_family[family]
        total = group["count"]
        executions = group["execution_successes"]
        rows.append(
            ASTCompatibilityRow(
                run_id=run.run_id,
                model=str(run.model),
                family=family,
                case_count=total,
                accepted_count=group["accepted"],
                acceptance_rate=group["accepted"] / total,
                acceptance_wilson_95=group["acceptance_wilson_95"],
                execution_count=executions,
                execution_rate=executions / total,
                execution_wilson_95=wilson_interval(successes=executions, total=total),
                rejection_categories=tuple(sorted(rejections.get(family, {}).items())),
            )
        )
    return rows


def build_plot_series(bundle: ValidatedStudyBundle) -> dict[str, Any]:
    """Return flat, plot-ready series derived from ``build_paper_tables``.

    Deliberately thin: it re-projects the same validated rows a plotting
    script would otherwise have to re-derive, so a chart and a table can never
    disagree. Each series is a list of flat dicts carrying the point estimate,
    its Wilson bounds, and the counts behind it -- never a rounded string.
    """
    tables = build_paper_tables(bundle)
    return {
        "native_protocol_validity": [
            {
                "configuration_id": row.configuration_id,
                "model": row.model,
                "setting": row.setting,
                "prompt_condition": row.prompt_condition,
                "count": row.protocol_valid_count,
                "denominator": row.case_count,
                "rate": row.protocol_valid_rate,
                "lower": row.protocol_valid_wilson_95["lower"],
                "upper": row.protocol_valid_wilson_95["upper"],
            }
            for row in tables.native_validity
        ],
        "adaptive_asr_by_monitor": [
            {
                "run_id": row.run_id,
                "family": row.family,
                "monitor": row.monitor,
                "count": row.target_effect_asr_count,
                "denominator": row.target_effect_asr_denominator,
                "rate": row.target_effect_asr,
                "lower": None
                if row.target_effect_asr_wilson_95 is None
                else row.target_effect_asr_wilson_95["lower"],
                "upper": None
                if row.target_effect_asr_wilson_95 is None
                else row.target_effect_asr_wilson_95["upper"],
            }
            for row in tables.adaptive_monitor
        ],
        "ast_acceptance_by_family": [
            {
                "run_id": row.run_id,
                "model": row.model,
                "family": row.family,
                "count": row.accepted_count,
                "denominator": row.case_count,
                "rate": row.acceptance_rate,
                "lower": row.acceptance_wilson_95["lower"],
                "upper": row.acceptance_wilson_95["upper"],
            }
            for row in tables.ast_compatibility
        ],
    }
