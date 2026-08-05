"""Cross-artifact validation and paper-ready aggregation for a study bundle.

A *study bundle* is the single, self-contained directory a Kaggle run exports
and a local machine imports. It is the only input from which manuscript
numbers may be generated: ``validate_study_bundle`` is the gate, and
``build_paper_tables`` / ``build_plot_series`` refuse to invent anything the
gate did not confirm.

Nothing in this repository produced such a directory before this module; the
schema below is normative and is what ``scripts/export_study_bundle.py`` must
write and ``scripts/import_study_bundle.py`` must read.

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
  "environment": "environment/....json", // REQUIRED when completed, else null
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
``_AST_RECORD_FIELDS`` / ``_ADAPTIVE_RECORD_FIELDS``). If those record
dataclasses gain or lose a field, this constant must be bumped in lockstep.

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
    verbatim benchmark/model-output fields in ``_VERBATIM_TEXT_KEYS``: that
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

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from secure_rag_bench.evaluation.adaptive_analysis import summarize_adaptive_records
from secure_rag_bench.evaluation.adaptive_runner import (
    AdaptiveRunRecord,
    MonitorConfiguration,
    RecordedToolCall,
)
from secure_rag_bench.evaluation.ast_compatibility import (
    ASTCompatibilityRecord,
    summarize_ast_compatibility,
)
from secure_rag_bench.evaluation.local_injecagent import (
    compute_execution_scores,
    compute_native_scores,
)
from secure_rag_bench.evaluation.native_analysis import (
    evaluate_validity_gate,
    summarize_validity,
    validated_run_identity,
    wilson_interval,
)
from secure_rag_bench.evaluation.native_monitor import NATIVE_DEFENSES
from secure_rag_bench.evaluation.study_artifacts import (
    StudyManifest,
    canonical_json,
    record_digest,
    redact_secrets,
)

BUNDLE_SCHEMA_VERSION = 1
SPLIT_SCHEMA_VERSION = 1
ENVIRONMENT_SCHEMA_VERSION = 1
AST_SCHEMA_VERSION = 1
ADAPTIVE_SCHEMA_VERSION = 1

BUNDLE_INDEX_NAME = "bundle.json"
FILE_MANIFEST_NAME = "MANIFEST.sha256"
REDACTION_SENTINEL = "[REDACTED]"

HELD_OUT_SIZE = 50
HELD_OUT_PER_ATTACK = 25

_STATUSES = frozenset({"completed", "skipped", "failed"})
_SETTINGS = frozenset({"base", "enhanced"})
_EXECUTION_KEYS = ("execution_step_1", "execution_step_2")
_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_AST_RECORD_FIELDS = frozenset(
    {"case_id", "family", "syntax_accepted", "execution_succeeded", "rejection_category", "detail"}
)
_ADAPTIVE_RECORD_FIELDS = frozenset(
    {
        "scenario_id",
        "family",
        "pair_id",
        "is_attack",
        "monitor",
        "retrieval_exposed",
        "plan_valid",
        "attempted_target_action",
        "halt_reason",
        "halt_category",
        "failed_property",
        "target_effect_executed",
        "benign_success",
        "retrieved_doc_ids",
        "recorded_calls",
        "internal_tool_calls",
    }
)

#: Fields holding verbatim benchmark or model text, exempt from the heuristic
#: token-shaped *value* scan. See the module docstring's point 12.
_VERBATIM_TEXT_KEYS = frozenset(
    {
        "system_prompt",
        "rendered_prompt",
        "rendered_prompt_step_2",
        "raw_output",
        "raw_output_step_2",
        "output",
        "output Step 2",
        "Tool Response",
        "Attacker Tool Response",
        "Attacker Instruction",
        "Attacker Tool Parameters",
        "Tool Parameters",
        "Thought",
        "Attacker Tool Thought",
        "detail",
        "halt_reason",
        "invalid_reason",
    }
)

_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}"),
)

_SECRET_PROBE = "__secure_rag_bench_secret_probe__"


class BundleValidationError(ValueError):
    """Raised when a study bundle cannot be trusted to produce paper numbers."""


# ---------------------------------------------------------------------------
# Validated bundle model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleModel:
    """One entry of the embedded ``configs/native_study_models.json`` catalog."""

    name: str
    provider: str
    model_id: str
    tier: str
    enabled: bool


@dataclass(frozen=True)
class SplitManifest:
    """A deterministic calibration/held-out case split."""

    split_id: str
    setting: str
    seed: int
    calibration: tuple[str, ...]
    held_out: tuple[str, ...]


@dataclass(frozen=True)
class NativeReplayArtifact:
    """One offline defense replay of a stage's saved trajectories."""

    defense: str
    records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class NativeStage:
    """One generation stage: its raw records, its run file, and its replays."""

    name: str
    records: tuple[Mapping[str, Any], ...]
    run_defense: str
    run_records: tuple[Mapping[str, Any], ...]
    gate: Any  # GateDecision, recomputed from ``records``
    expected_case_ids: tuple[str, ...]
    run_identity: Mapping[str, Any]
    replays: tuple[NativeReplayArtifact, ...]


@dataclass(frozen=True)
class NativeConfiguration:
    """One model/setting/prompt configuration and everything it produced."""

    configuration_id: str
    model: str
    tier: str
    model_id: str
    model_revision: str | None
    dataset_revision: str | None
    setting: str
    prompt_condition: str
    status: str
    status_reason: str | None
    split_id: str
    environment: Mapping[str, Any] | None
    pilot: NativeStage | None
    full: NativeStage | None


@dataclass(frozen=True)
class ASTRun:
    """One restricted-AST compatibility run and its recomputed summary."""

    run_id: str
    model: str | None
    status: str
    status_reason: str | None
    records: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class AdaptiveRun:
    """One adaptive attack/control sweep and its recomputed summary."""

    run_id: str
    status: str
    status_reason: str | None
    records: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


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

    file_digests = _validate_file_manifest(root_path)
    index = _load_json(root_path, BUNDLE_INDEX_NAME)
    _require_int(index, "schema_version", BUNDLE_INDEX_NAME)
    if index["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise BundleValidationError(
            f"{BUNDLE_INDEX_NAME}: unsupported schema_version "
            f"{index['schema_version']!r}; expected {BUNDLE_SCHEMA_VERSION}"
        )
    study_id = _require_text(index, "study_id", BUNDLE_INDEX_NAME)
    created_utc = _require_timestamp(index, "created_utc", BUNDLE_INDEX_NAME)

    models = _parse_models(index)
    splits = _load_splits(root_path)
    split_ids = {split.split_id for split in splits}

    configurations = tuple(
        _validate_configuration(root_path, entry, models, splits, index_position)
        for index_position, entry in enumerate(_require_list(index, "configurations", BUNDLE_INDEX_NAME))
    )
    _require_unique(
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
        _validate_ast_run(root_path, entry, models)
        for entry in _require_list(index, "ast_runs", BUNDLE_INDEX_NAME)
    )
    adaptive_runs = tuple(
        _validate_adaptive_run(root_path, entry)
        for entry in _require_list(index, "adaptive_runs", BUNDLE_INDEX_NAME)
    )
    _require_unique((run.run_id for run in ast_runs), label="ast run_id")
    _require_unique((run.run_id for run in adaptive_runs), label="adaptive run_id")

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
# File-level integrity
# ---------------------------------------------------------------------------


def _validate_file_manifest(root: Path) -> dict[str, str]:
    manifest_path = root / FILE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleValidationError(f"{FILE_MANIFEST_NAME}: missing bundle file manifest")
    digests: dict[str, str] = {}
    for number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or not _DIGEST_PATTERN.match(digest):
            raise BundleValidationError(
                f"{FILE_MANIFEST_NAME}: line {number} is not '<sha256>  <path>'"
            )
        relative = relative.strip()
        _require_safe_relative_path(relative, FILE_MANIFEST_NAME)
        if relative in digests:
            raise BundleValidationError(
                f"{FILE_MANIFEST_NAME}: duplicate entry for {relative}"
            )
        digests[relative] = digest

    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != FILE_MANIFEST_NAME
    }
    for relative in sorted(set(digests) - present):
        raise BundleValidationError(f"{relative}: listed in {FILE_MANIFEST_NAME} but absent")
    for relative in sorted(present - set(digests)):
        raise BundleValidationError(f"{relative}: present but missing from {FILE_MANIFEST_NAME}")
    for relative, digest in sorted(digests.items()):
        actual = sha256((root / relative).read_bytes()).hexdigest()
        if actual != digest:
            raise BundleValidationError(
                f"{relative}: file sha256 mismatch ({actual} != {digest})"
            )
    return digests


def _require_safe_relative_path(relative: str, label: str) -> None:
    if not relative:
        raise BundleValidationError(f"{label}: empty path")
    if relative != relative.strip() or "\\" in relative:
        raise BundleValidationError(f"{label}: {relative!r} must be a POSIX relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise BundleValidationError(f"{label}: {relative!r} escapes the bundle root")


# ---------------------------------------------------------------------------
# Loading primitives
# ---------------------------------------------------------------------------


def _load_json(root: Path, relative: str) -> dict[str, Any]:
    _require_safe_relative_path(relative, BUNDLE_INDEX_NAME)
    path = root / relative
    if not path.is_file():
        raise BundleValidationError(f"{relative}: declared file is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"{relative}: unreadable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BundleValidationError(f"{relative}: expected a JSON object")
    _assert_no_unredacted_secrets(payload, label=relative)
    return payload


def _load_checkpoint_records(root: Path, relative: str) -> list[dict[str, Any]]:
    """Validate one JSONL checkpoint's envelopes without mutating the bundle.

    Mirrors ``JsonlCheckpointStore``'s envelope contract exactly, but never
    locks or rewrites the file -- see the module docstring's closing note.
    """
    _require_safe_relative_path(relative, BUNDLE_INDEX_NAME)
    path = root / relative
    if not path.is_file():
        raise BundleValidationError(f"{relative}: declared checkpoint is missing")
    raw = path.read_bytes()
    if not raw:
        raise BundleValidationError(f"{relative}: checkpoint is empty")
    if not raw.endswith(b"\n"):
        raise BundleValidationError(f"{relative}: checkpoint has an unterminated final line")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line:
            raise BundleValidationError(f"{relative}: blank checkpoint line {number}")
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BundleValidationError(f"{relative}: invalid JSON on line {number}: {exc}") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"case_id", "payload", "sha256"}:
            raise BundleValidationError(f"{relative}: invalid envelope on line {number}")
        case_id = envelope["case_id"]
        payload = envelope["payload"]
        if not isinstance(case_id, str) or not case_id or not isinstance(payload, dict):
            raise BundleValidationError(f"{relative}: invalid envelope fields on line {number}")
        if payload.get("case_id") != case_id:
            raise BundleValidationError(f"{relative}: case_id mismatch on line {number}")
        if case_id in seen:
            raise BundleValidationError(f"{relative}: duplicate case_id {case_id!r} on line {number}")
        seen.add(case_id)
        actual = record_digest(payload)
        if actual != envelope["sha256"]:
            raise BundleValidationError(
                f"{relative}: envelope sha256 mismatch on line {number} "
                f"({actual} != {envelope['sha256']})"
            )
        _assert_no_unredacted_secrets(payload, label=f"{relative}:{number}")
        _validate_record_digest(payload, label=f"{relative}:{number}")
        records.append(payload)
    return records


def _validate_record_digest(record: Mapping[str, Any], *, label: str) -> None:
    claimed = record.get("record_sha256")
    if not isinstance(claimed, str) or not _DIGEST_PATTERN.match(claimed):
        raise BundleValidationError(f"{label}: record is missing a valid record_sha256")
    unhashed = {key: value for key, value in record.items() if key != "record_sha256"}
    actual = record_digest(unhashed)
    if actual != claimed:
        raise BundleValidationError(
            f"{label}: record_sha256 mismatch for case {record.get('case_id')!r} "
            f"({actual} != {claimed})"
        )


# ---------------------------------------------------------------------------
# Secret scanning
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _is_secret_key(key: str) -> bool:
    """Whether ``redact_secrets`` would replace this key's value.

    Probed through ``redact_secrets`` itself rather than duplicating its
    keyword list, so the two can never disagree about what counts as a
    credential-shaped key.
    """
    return redact_secrets({key: _SECRET_PROBE}).get(key) == REDACTION_SENTINEL


def _assert_no_unredacted_secrets(payload: Any, *, label: str) -> None:
    """Reject credential-shaped keys and token-shaped values anywhere inside."""
    stack: list[tuple[str, Any, bool]] = [("", payload, False)]
    while stack:
        path, node, verbatim = stack.pop()
        if isinstance(node, Mapping):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                if isinstance(key, str) and _is_secret_key(key) and value != REDACTION_SENTINEL:
                    raise BundleValidationError(
                        f"{label}: credential-shaped key {child!r} is not redacted"
                    )
                stack.append((child, value, isinstance(key, str) and key in _VERBATIM_TEXT_KEYS))
        elif isinstance(node, list):
            for position, item in enumerate(node):
                stack.append((f"{path}[{position}]", item, verbatim))
        elif isinstance(node, str) and not verbatim:
            for pattern in _SECRET_VALUE_PATTERNS:
                if pattern.search(node):
                    raise BundleValidationError(
                        f"{label}: token-shaped secret value at {path or '<root>'!r}"
                    )


# ---------------------------------------------------------------------------
# Index parsing helpers
# ---------------------------------------------------------------------------


def _require_list(payload: Mapping[str, Any], key: str, label: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise BundleValidationError(f"{label}: {key!r} must be a list")
    return value


def _require_text(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BundleValidationError(f"{label}: {key!r} must be a non-empty string")
    return value


def _require_int(payload: Mapping[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BundleValidationError(f"{label}: {key!r} must be an integer")
    return value


def _require_timestamp(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = _require_text(payload, key, label)
    if not _TIMESTAMP_PATTERN.match(value):
        raise BundleValidationError(
            f"{label}: {key!r} must be an ISO-8601 UTC timestamp ending in 'Z'"
        )
    return value


def _require_unique(values: Iterable[str], *, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise BundleValidationError(f"{BUNDLE_INDEX_NAME}: duplicate {label} {value!r}")
        seen.add(value)


def _parse_models(index: Mapping[str, Any]) -> tuple[BundleModel, ...]:
    entries = _require_list(index, "models", BUNDLE_INDEX_NAME)
    if not entries:
        raise BundleValidationError(f"{BUNDLE_INDEX_NAME}: 'models' must not be empty")
    models: list[BundleModel] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise BundleValidationError(f"{BUNDLE_INDEX_NAME}: each model must be an object")
        enabled = entry.get("enabled")
        if not isinstance(enabled, bool):
            raise BundleValidationError(f"{BUNDLE_INDEX_NAME}: model 'enabled' must be a boolean")
        models.append(
            BundleModel(
                name=_require_text(entry, "name", BUNDLE_INDEX_NAME),
                provider=_require_text(entry, "provider", BUNDLE_INDEX_NAME),
                model_id=_require_text(entry, "model_id", BUNDLE_INDEX_NAME),
                tier=_require_text(entry, "tier", BUNDLE_INDEX_NAME),
                enabled=enabled,
            )
        )
    _require_unique((model.name for model in models), label="model name")
    return tuple(models)


def _load_splits(root: Path) -> tuple[SplitManifest, ...]:
    directory = root / "splits"
    if not directory.is_dir():
        raise BundleValidationError("splits/: bundle must contain at least one split manifest")
    splits: list[SplitManifest] = []
    for path in sorted(directory.glob("*.json")):
        relative = path.relative_to(root).as_posix()
        payload = _load_json(root, relative)
        if _require_int(payload, "schema_version", relative) != SPLIT_SCHEMA_VERSION:
            raise BundleValidationError(f"{relative}: unsupported split schema_version")
        setting = _require_text(payload, "setting", relative)
        if setting not in _SETTINGS:
            raise BundleValidationError(f"{relative}: setting must be 'base' or 'enhanced'")
        held_out = _string_list(payload, "held_out", relative)
        calibration = _string_list(payload, "calibration", relative)
        if len(set(held_out)) != len(held_out):
            raise BundleValidationError(f"{relative}: held_out contains duplicate case ids")
        if len(set(calibration)) != len(calibration):
            raise BundleValidationError(f"{relative}: calibration contains duplicate case ids")
        overlap = set(held_out) & set(calibration)
        if overlap:
            raise BundleValidationError(
                f"{relative}: held_out and calibration share {len(overlap)} case id(s)"
            )
        if len(held_out) != HELD_OUT_SIZE:
            raise BundleValidationError(
                f"{relative}: held_out must contain exactly {HELD_OUT_SIZE} case ids, "
                f"found {len(held_out)}"
            )
        splits.append(
            SplitManifest(
                split_id=path.stem,
                setting=setting,
                seed=_require_int(payload, "seed", relative),
                calibration=tuple(calibration),
                held_out=tuple(held_out),
            )
        )
    if not splits:
        raise BundleValidationError("splits/: bundle must contain at least one split manifest")
    return tuple(splits)


def _string_list(payload: Mapping[str, Any], key: str, label: str) -> list[str]:
    values = _require_list(payload, key, label)
    if any(not isinstance(value, str) or not value for value in values):
        raise BundleValidationError(f"{label}: {key!r} must contain non-empty strings")
    return list(values)


# ---------------------------------------------------------------------------
# Native configuration validation
# ---------------------------------------------------------------------------


def _validate_configuration(
    root: Path,
    entry: Any,
    models: Sequence[BundleModel],
    splits: Sequence[SplitManifest],
    position: int,
) -> NativeConfiguration:
    label = f"{BUNDLE_INDEX_NAME}: configurations[{position}]"
    if not isinstance(entry, Mapping):
        raise BundleValidationError(f"{label} must be an object")
    configuration_id = _require_text(entry, "configuration_id", label)
    label = f"{BUNDLE_INDEX_NAME}: configuration {configuration_id!r}"

    model_name = _require_text(entry, "model", label)
    catalog = next((model for model in models if model.name == model_name), None)
    if catalog is None:
        raise BundleValidationError(f"{label} names unknown model {model_name!r}")
    tier = _require_text(entry, "tier", label)
    if tier != catalog.tier:
        raise BundleValidationError(
            f"{label} declares tier {tier!r} but the catalog says {catalog.tier!r}"
        )
    setting = _require_text(entry, "setting", label)
    if setting not in _SETTINGS:
        raise BundleValidationError(f"{label} setting must be 'base' or 'enhanced'")
    prompt_condition = _require_text(entry, "prompt_condition", label)
    split_id = _require_text(entry, "split", label)
    split = next((item for item in splits if item.split_id == split_id), None)

    status = _require_text(entry, "status", label)
    if status not in _STATUSES:
        raise BundleValidationError(
            f"{label} status must be one of {sorted(_STATUSES)}, found {status!r}"
        )
    status_reason = entry.get("status_reason")
    if status == "completed":
        if status_reason not in (None, ""):
            raise BundleValidationError(f"{label} completed configuration must not carry a reason")
        status_reason = None
    elif not isinstance(status_reason, str) or not status_reason.strip():
        raise BundleValidationError(
            f"{label} status {status!r} requires a non-empty 'status_reason'"
        )

    if status != "completed" and entry.get("full") is not None:
        raise BundleValidationError(
            f"{label} status {status!r} must not carry a full native stage"
        )

    pilot = (
        _validate_stage(root, entry["pilot"], label=f"{label} pilot", is_pilot=True)
        if entry.get("pilot") is not None
        else None
    )
    full = (
        _validate_stage(root, entry["full"], label=f"{label} full", is_pilot=False)
        if entry.get("full") is not None
        else None
    )

    if status == "completed":
        if pilot is None or full is None:
            raise BundleValidationError(
                f"{label} completed configuration requires both a pilot and a full stage"
            )
        if not pilot.gate.passed:
            raise BundleValidationError(
                f"{label} claims status 'completed' but its recomputed held-out gate "
                f"failed: {list(pilot.gate.reasons)}"
            )

    if split is not None and pilot is not None:
        _validate_split_membership(pilot, split, label=f"{label} pilot", is_pilot=True)
    if split is not None and full is not None:
        _validate_split_membership(full, split, label=f"{label} full", is_pilot=False)

    identities = [stage.run_identity for stage in (pilot, full) if stage is not None]
    model_revision: str | None = None
    dataset_revision: str | None = None
    for identity, stage_label in zip(identities, ("pilot", "full")):
        _validate_identity(
            identity,
            catalog=catalog,
            setting=setting,
            prompt_condition=prompt_condition,
            label=f"{label} {stage_label}",
        )
        model_revision = str(identity["model"]["model_revision"])
        dataset_revision = str(identity["dataset_revision"])
    if len({str(identity["model"]["model_revision"]) for identity in identities}) > 1:
        raise BundleValidationError(f"{label} stages disagree about the model revision")
    if len({str(identity["dataset_revision"]) for identity in identities}) > 1:
        raise BundleValidationError(f"{label} stages disagree about the dataset revision")

    environment_path = entry.get("environment")
    environment: Mapping[str, Any] | None = None
    if environment_path is not None:
        if not isinstance(environment_path, str):
            raise BundleValidationError(f"{label} 'environment' must be a string path or null")
        environment = _validate_environment(root, environment_path, catalog)
    elif status == "completed":
        raise BundleValidationError(
            f"{label} completed configuration requires an 'environment' capture"
        )

    return NativeConfiguration(
        configuration_id=configuration_id,
        model=model_name,
        tier=tier,
        model_id=catalog.model_id,
        model_revision=model_revision,
        dataset_revision=dataset_revision,
        setting=setting,
        prompt_condition=prompt_condition,
        status=status,
        status_reason=status_reason,
        split_id=split_id,
        environment=environment,
        pilot=pilot,
        full=full,
    )


def _validate_stage(root: Path, entry: Any, *, label: str, is_pilot: bool) -> NativeStage:
    if not isinstance(entry, Mapping):
        raise BundleValidationError(f"{label}: stage must be an object")
    checkpoint_path = _require_text(entry, "checkpoint", label)
    run_path = _require_text(entry, "run", label)
    manifest_path = _require_text(entry, "manifest", label)

    records = _load_checkpoint_records(root, checkpoint_path)
    records_by_id = {str(record["case_id"]): record for record in records}
    try:
        identity = validated_run_identity(records)
    except (TypeError, ValueError) as exc:
        raise BundleValidationError(
            f"{checkpoint_path}: inconsistent run identity or run_identity_sha256: {exc}"
        ) from exc

    expected_ids = _validate_stage_manifest(root, manifest_path, records_by_id)
    missing = [case_id for case_id in expected_ids if case_id not in records_by_id]
    if missing:
        raise BundleValidationError(
            f"{checkpoint_path}: {len(missing)} expected case(s) missing, first {missing[0]!r}"
        )
    if is_pilot:
        _validate_pilot_balance(records, label=checkpoint_path)

    gate = evaluate_validity_gate(records)
    run_defense, run_records = _validate_run_file(
        root, run_path, records_by_id, gate, expect_replay_only=False, label=label
    )

    replays: list[NativeReplayArtifact] = []
    for replay_path in _replay_paths(entry, label):
        defense, replay_records = _validate_run_file(
            root, replay_path, records_by_id, gate, expect_replay_only=True, label=label
        )
        replays.append(NativeReplayArtifact(defense=defense, records=tuple(replay_records)))
    _require_unique((replay.defense for replay in replays), label=f"{label} replay defense")

    return NativeStage(
        name="pilot" if is_pilot else "full",
        records=tuple(records),
        run_defense=run_defense,
        run_records=tuple(run_records),
        gate=gate,
        expected_case_ids=tuple(expected_ids),
        run_identity=identity,
        replays=tuple(replays),
    )


def _replay_paths(entry: Mapping[str, Any], label: str) -> list[str]:
    replays = entry.get("replays", [])
    if not isinstance(replays, list) or any(not isinstance(item, str) for item in replays):
        raise BundleValidationError(f"{label}: 'replays' must be a list of paths")
    return list(replays)


def _validate_stage_manifest(
    root: Path, relative: str, records_by_id: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    envelope = _load_json(root, relative)
    if set(envelope) != {"payload", "sha256"}:
        raise BundleValidationError(f"{relative}: expected a {{'payload','sha256'}} envelope")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise BundleValidationError(f"{relative}: manifest payload must be an object")
    expected_ids = _string_list(payload, "expected_case_ids", relative)
    recomputed = StudyManifest.from_records(dict(records_by_id), expected_case_ids=expected_ids)
    if recomputed.sha256 != envelope["sha256"]:
        raise BundleValidationError(
            f"{relative}: manifest sha256 mismatch ({recomputed.sha256} != {envelope['sha256']})"
        )
    if canonical_json(recomputed.payload) != canonical_json(payload):
        raise BundleValidationError(
            f"{relative}: manifest payload does not describe the checkpoint records"
        )
    return expected_ids


def _validate_pilot_balance(records: Sequence[Mapping[str, Any]], *, label: str) -> None:
    counts = {"dh": 0, "ds": 0}
    for record in records:
        attack = record.get("attack")
        if attack not in counts:
            raise BundleValidationError(f"{label}: record has unknown attack class {attack!r}")
        counts[str(attack)] += 1
    if counts != {"dh": HELD_OUT_PER_ATTACK, "ds": HELD_OUT_PER_ATTACK}:
        raise BundleValidationError(
            f"{label}: held-out pilot must be exactly {HELD_OUT_PER_ATTACK} direct-harm and "
            f"{HELD_OUT_PER_ATTACK} data-stealing cases, found {counts}"
        )


def _validate_run_file(
    root: Path,
    relative: str,
    records_by_id: Mapping[str, Mapping[str, Any]],
    gate: Any,
    *,
    expect_replay_only: bool,
    label: str,
) -> tuple[str, list[Mapping[str, Any]]]:
    payload = _load_json(root, relative)
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise BundleValidationError(f"{relative}: missing 'protocol' object")
    defense = _require_text(protocol, "defense", relative)
    if defense not in NATIVE_DEFENSES:
        raise BundleValidationError(f"{relative}: unsupported defense {defense!r}")
    if protocol.get("replay_only") is not expect_replay_only:
        raise BundleValidationError(
            f"{relative}: 'replay_only' must be {expect_replay_only} for this artifact"
        )

    derived = payload.get("records")
    if not isinstance(derived, list) or not derived:
        raise BundleValidationError(f"{relative}: 'records' must be a non-empty list")
    _validate_derived_records(records_by_id, derived, label=relative)
    if protocol.get("case_count") != len(derived):
        raise BundleValidationError(
            f"{relative}: protocol case_count {protocol.get('case_count')!r} does not match "
            f"{len(derived)} records"
        )

    source = [dict(record) for record in records_by_id.values()]
    if canonical_json({"scores": compute_native_scores(source)}) != canonical_json(
        {"scores": payload.get("scores")}
    ):
        raise BundleValidationError(
            f"{relative}: 'scores' do not recompute from the raw checkpoint records"
        )
    if canonical_json(
        {"scores": compute_execution_scores([dict(record) for record in derived])}
    ) != canonical_json({"scores": payload.get("execution_scores")}):
        raise BundleValidationError(
            f"{relative}: 'execution_scores' do not recompute from the replayed records"
        )

    claimed_gate = payload.get("gate_decision")
    if not isinstance(claimed_gate, Mapping):
        raise BundleValidationError(f"{relative}: missing 'gate_decision'")
    if canonical_json(_gate_to_payload(gate)) != canonical_json(dict(claimed_gate)):
        raise BundleValidationError(
            f"{relative}: claimed gate_decision does not match the gate recomputed from "
            f"the raw records"
        )
    return defense, list(derived)


def _gate_to_payload(gate: Any) -> dict[str, Any]:
    return {
        "passed": gate.passed,
        "reasons": list(gate.reasons),
        "required_rate": gate.required_rate,
        "protocol_valid_count": gate.protocol_valid_count,
        "protocol_valid_denominator": gate.protocol_valid_denominator,
        "protocol_valid_rate": gate.protocol_valid_rate,
        "wilson_95": dict(gate.wilson_95),
        "attack_counts": dict(gate.attack_counts),
    }


def _validate_derived_records(
    records_by_id: Mapping[str, Mapping[str, Any]],
    derived: Sequence[Any],
    *,
    label: str,
) -> None:
    """Every run/replay record must be its source record plus execution keys."""
    seen: set[str] = set()
    for position, record in enumerate(derived):
        if not isinstance(record, Mapping):
            raise BundleValidationError(f"{label}: records[{position}] must be an object")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or case_id not in records_by_id:
            raise BundleValidationError(
                f"{label}: records[{position}] case_id {case_id!r} is not in the checkpoint"
            )
        if case_id in seen:
            raise BundleValidationError(f"{label}: duplicate case_id {case_id!r} in records")
        seen.add(case_id)
        stripped = {key: value for key, value in record.items() if key not in _EXECUTION_KEYS}
        _validate_record_digest(stripped, label=f"{label}:{case_id}")
        if canonical_json(stripped) != canonical_json(dict(records_by_id[case_id])):
            raise BundleValidationError(
                f"{label}: record {case_id!r} differs from its checkpoint source beyond "
                f"replayed execution decisions"
            )
        for key in _EXECUTION_KEYS:
            decision = record.get(key)
            if decision is not None and (
                not isinstance(decision, Mapping) or not isinstance(decision.get("allowed"), bool)
            ):
                raise BundleValidationError(f"{label}: record {case_id!r} has an invalid {key}")
    missing = sorted(set(records_by_id) - seen)
    if missing:
        raise BundleValidationError(
            f"{label}: {len(missing)} checkpoint case(s) absent from records, "
            f"first {missing[0]!r}"
        )


def _validate_split_membership(
    stage: NativeStage, split: SplitManifest, *, label: str, is_pilot: bool
) -> None:
    stage_ids = {str(record["case_id"]) for record in stage.records}
    if is_pilot:
        if stage_ids != set(split.held_out):
            raise BundleValidationError(
                f"{label}: pilot cases must be exactly the split's held_out set"
            )
        return
    for name, expected in (("held_out", split.held_out), ("calibration", split.calibration)):
        missing = sorted(set(expected) - stage_ids)
        if missing:
            raise BundleValidationError(
                f"{label}: full run is missing {len(missing)} {name} case(s), "
                f"first {missing[0]!r}"
            )


def _validate_identity(
    identity: Mapping[str, Any],
    *,
    catalog: BundleModel,
    setting: str,
    prompt_condition: str,
    label: str,
) -> None:
    model = identity.get("model")
    if not isinstance(model, Mapping):
        raise BundleValidationError(f"{label}: run identity has no model block")
    model_id = model.get("model_id")
    if model_id != catalog.model_id:
        raise BundleValidationError(
            f"{label}: records were generated with model_id {model_id!r}, but the catalog "
            f"declares {catalog.model_id!r}"
        )
    revision = model.get("model_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise BundleValidationError(
            f"{label}: records carry no model_revision; pin an immutable revision (a dated "
            f"provider snapshot for hosted endpoints) before exporting"
        )
    dataset_revision = identity.get("dataset_revision")
    if not isinstance(dataset_revision, str) or dataset_revision.strip() in {"", "unknown"}:
        raise BundleValidationError(f"{label}: records carry no usable dataset_revision")
    if identity.get("setting") != setting:
        raise BundleValidationError(
            f"{label}: records were generated for setting {identity.get('setting')!r}, "
            f"not {setting!r}"
        )
    if identity.get("prompt_condition") != prompt_condition:
        raise BundleValidationError(
            f"{label}: records were generated for prompt condition "
            f"{identity.get('prompt_condition')!r}, not {prompt_condition!r}"
        )


def _validate_environment(
    root: Path, relative: str, catalog: BundleModel
) -> Mapping[str, Any]:
    payload = _load_json(root, relative)
    if _require_int(payload, "schema_version", relative) != ENVIRONMENT_SCHEMA_VERSION:
        raise BundleValidationError(f"{relative}: unsupported environment schema_version")
    _require_timestamp(payload, "captured_utc", relative)

    platform = payload.get("platform")
    if not isinstance(platform, Mapping):
        raise BundleValidationError(f"{relative}: 'platform' must be an object")
    _require_text(platform, "python_version", relative)

    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping) or not isinstance(gpu.get("available"), bool):
        raise BundleValidationError(f"{relative}: 'gpu.available' must be a boolean")
    if gpu["available"]:
        devices = _require_list(gpu, "devices", relative)
        if not devices:
            raise BundleValidationError(f"{relative}: 'gpu.devices' must not be empty")
        for device in devices:
            if not isinstance(device, Mapping):
                raise BundleValidationError(f"{relative}: each GPU device must be an object")
            _require_text(device, "name", relative)

    packages = payload.get("packages")
    if not isinstance(packages, Mapping) or not packages:
        raise BundleValidationError(f"{relative}: 'packages' must be a non-empty object")
    if any(not isinstance(version, str) or not version for version in packages.values()):
        raise BundleValidationError(f"{relative}: package versions must be non-empty strings")
    if catalog.provider == "transformers":
        for required in ("torch", "transformers"):
            if required not in packages:
                raise BundleValidationError(
                    f"{relative}: a transformers-backed configuration must pin {required!r}"
                )
    return payload


# ---------------------------------------------------------------------------
# AST and adaptive run validation
# ---------------------------------------------------------------------------


def _validate_run_entry(entry: Any, *, kind: str) -> tuple[str, str, str | None, str | None]:
    if not isinstance(entry, Mapping):
        raise BundleValidationError(f"{BUNDLE_INDEX_NAME}: each {kind} run must be an object")
    run_id = _require_text(entry, "run_id", BUNDLE_INDEX_NAME)
    label = f"{BUNDLE_INDEX_NAME}: {kind} run {run_id!r}"
    status = _require_text(entry, "status", label)
    if status not in _STATUSES:
        raise BundleValidationError(f"{label} status must be one of {sorted(_STATUSES)}")
    status_reason = entry.get("status_reason")
    if status == "completed":
        status_reason = None
    elif not isinstance(status_reason, str) or not status_reason.strip():
        raise BundleValidationError(f"{label} status {status!r} requires a 'status_reason'")
    path = entry.get("path")
    if status == "completed" and (not isinstance(path, str) or not path):
        raise BundleValidationError(f"{label} completed run requires a 'path'")
    return run_id, status, status_reason, path if isinstance(path, str) else None


def _validate_schema_version(entry: Mapping[str, Any], expected: int, label: str) -> None:
    if _require_int(entry, "schema_version", label) != expected:
        raise BundleValidationError(
            f"{label}: unsupported record schema_version "
            f"{entry['schema_version']!r}; expected {expected}"
        )


def _validate_ast_run(root: Path, entry: Any, models: Sequence[BundleModel]) -> ASTRun:
    run_id, status, status_reason, path = _validate_run_entry(entry, kind="ast")
    label = f"{BUNDLE_INDEX_NAME}: ast run {run_id!r}"
    assert isinstance(entry, Mapping)
    _validate_schema_version(entry, AST_SCHEMA_VERSION, label)
    model = entry.get("model")
    if status == "completed":
        if not isinstance(model, str) or model not in {item.name for item in models}:
            raise BundleValidationError(f"{label} names unknown model {model!r}")
    if status != "completed" or path is None:
        return ASTRun(
            run_id=run_id,
            model=model if isinstance(model, str) else None,
            status=status,
            status_reason=status_reason,
            records=(),
            summary={},
        )

    payload = _load_json(root, path)
    records = _require_list(payload, "records", path)
    if not records:
        raise BundleValidationError(f"{path}: 'records' must not be empty")
    parsed: list[ASTCompatibilityRecord] = []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != _AST_RECORD_FIELDS:
            raise BundleValidationError(
                f"{path}: records[{position}] does not match AST schema version "
                f"{AST_SCHEMA_VERSION}"
            )
        try:
            parsed.append(ASTCompatibilityRecord(**dict(record)))
        except TypeError as exc:
            raise BundleValidationError(f"{path}: records[{position}] is malformed: {exc}") from exc

    _validate_embedded_manifest(
        payload,
        keyed={f"{record['case_id']}::{position}": dict(record) for position, record in enumerate(records)},
        label=path,
    )
    recomputed = summarize_ast_compatibility(parsed)
    if canonical_json(recomputed) != canonical_json(_require_mapping(payload, "summary", path)):
        raise BundleValidationError(f"{path}: 'summary' does not recompute from its records")
    return ASTRun(
        run_id=run_id,
        model=str(model),
        status=status,
        status_reason=status_reason,
        records=tuple(dict(record) for record in records),
        summary=recomputed,
    )


def _validate_adaptive_run(root: Path, entry: Any) -> AdaptiveRun:
    run_id, status, status_reason, path = _validate_run_entry(entry, kind="adaptive")
    label = f"{BUNDLE_INDEX_NAME}: adaptive run {run_id!r}"
    assert isinstance(entry, Mapping)
    _validate_schema_version(entry, ADAPTIVE_SCHEMA_VERSION, label)
    if status != "completed" or path is None:
        return AdaptiveRun(
            run_id=run_id, status=status, status_reason=status_reason, records=(), summary={}
        )

    payload = _load_json(root, path)
    records = _require_list(payload, "records", path)
    if not records:
        raise BundleValidationError(f"{path}: 'records' must not be empty")
    parsed = [
        _parse_adaptive_record(record, position, path) for position, record in enumerate(records)
    ]
    _validate_embedded_manifest(
        payload,
        keyed={
            f"{record['scenario_id']}::{record['monitor']}": dict(record) for record in records
        },
        label=path,
    )
    recomputed = summarize_adaptive_records(parsed)
    if canonical_json(recomputed) != canonical_json(_require_mapping(payload, "summary", path)):
        raise BundleValidationError(f"{path}: 'summary' does not recompute from its records")
    return AdaptiveRun(
        run_id=run_id,
        status=status,
        status_reason=status_reason,
        records=tuple(dict(record) for record in records),
        summary=recomputed,
    )


def _parse_adaptive_record(record: Any, position: int, label: str) -> AdaptiveRunRecord:
    """Rebuild one ``AdaptiveRunRecord`` from its serialized flat shape."""
    if not isinstance(record, Mapping) or set(record) != _ADAPTIVE_RECORD_FIELDS:
        raise BundleValidationError(
            f"{label}: records[{position}] does not match adaptive schema version "
            f"{ADAPTIVE_SCHEMA_VERSION}"
        )
    try:
        monitor = MonitorConfiguration(record["monitor"])
    except ValueError as exc:
        raise BundleValidationError(
            f"{label}: records[{position}] has unknown monitor {record['monitor']!r}"
        ) from exc
    calls = record["recorded_calls"]
    if not isinstance(calls, list):
        raise BundleValidationError(f"{label}: records[{position}] 'recorded_calls' must be a list")
    recorded = tuple(
        RecordedToolCall(
            tool=str(call["tool"]), args=tuple(call["args"]), kwargs=dict(call["kwargs"])
        )
        for call in calls
        if isinstance(call, Mapping)
    )
    if len(recorded) != len(calls):
        raise BundleValidationError(f"{label}: records[{position}] has a malformed recorded call")
    parsed = AdaptiveRunRecord(
        scenario_id=str(record["scenario_id"]),
        family=str(record["family"]),
        pair_id=str(record["pair_id"]),
        is_attack=bool(record["is_attack"]),
        monitor=monitor,
        retrieval_exposed=bool(record["retrieval_exposed"]),
        plan_valid=bool(record["plan_valid"]),
        attempted_target_action=bool(record["attempted_target_action"]),
        halt_reason=record["halt_reason"],
        halt_category=record["halt_category"],
        failed_property=record["failed_property"],
        target_effect_executed=bool(record["target_effect_executed"]),
        benign_success=bool(record["benign_success"]),
        retrieved_doc_ids=tuple(record["retrieved_doc_ids"]),
        recorded_calls=recorded,
        internal_tool_calls=tuple(record["internal_tool_calls"]),
    )
    if parsed.benign_success != (parsed.target_effect_executed and not parsed.is_attack):
        raise BundleValidationError(
            f"{label}: records[{position}] 'benign_success' contradicts "
            f"'target_effect_executed'/'is_attack'"
        )
    return parsed


def _validate_embedded_manifest(
    payload: Mapping[str, Any], *, keyed: Mapping[str, Mapping[str, Any]], label: str
) -> None:
    envelope = payload.get("manifest")
    if not isinstance(envelope, Mapping) or set(envelope) != {"payload", "sha256"}:
        raise BundleValidationError(f"{label}: 'manifest' must be a {{'payload','sha256'}} envelope")
    manifest_payload = envelope["payload"]
    if not isinstance(manifest_payload, Mapping):
        raise BundleValidationError(f"{label}: manifest payload must be an object")
    expected_ids = _string_list(manifest_payload, "expected_case_ids", label)
    recomputed = StudyManifest.from_records(dict(keyed), expected_case_ids=expected_ids)
    if recomputed.sha256 != envelope["sha256"]:
        raise BundleValidationError(
            f"{label}: manifest sha256 mismatch ({recomputed.sha256} != {envelope['sha256']})"
        )
    if canonical_json(recomputed.payload) != canonical_json(dict(manifest_payload)):
        raise BundleValidationError(f"{label}: manifest payload does not describe its records")
    missing = [key for key in expected_ids if key not in keyed]
    if missing:
        raise BundleValidationError(
            f"{label}: {len(missing)} expected record(s) missing, first {missing[0]!r}"
        )


def _require_mapping(payload: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BundleValidationError(f"{label}: {key!r} must be an object")
    return value


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
