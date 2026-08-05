"""Artifact-specific validation chains for a study bundle.

This is the middle layer between ``study_bundle_io`` (generic integrity and
JSON-shape primitives) and ``study_reporting`` (the public entry point and the
paper tables). It knows what a native generation stage, a restricted-AST run,
and an adaptive sweep are *supposed* to look like, and it produces the parsed,
trusted dataclasses that ``ValidatedStudyBundle`` is assembled from.

See ``study_reporting``'s module docstring for the normative bundle schema and
the full list of checks; this module is where checks 2-11 of that list live.

Nothing here is part of the public surface -- import from ``study_reporting``
instead, which re-exports everything a caller needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    validated_run_identity,
)
from secure_rag_bench.evaluation.native_monitor import NATIVE_DEFENSES
from secure_rag_bench.evaluation.study_artifacts import StudyManifest, canonical_json
from secure_rag_bench.evaluation.study_bundle_io import (
    BUNDLE_INDEX_NAME,
    BundleValidationError,
    load_checkpoint_records,
    load_json,
    require_int,
    require_list,
    require_mapping,
    require_text,
    require_timestamp,
    require_unique,
    string_list,
    validate_record_digest,
)

BUNDLE_SCHEMA_VERSION = 1
SPLIT_SCHEMA_VERSION = 1
ENVIRONMENT_SCHEMA_VERSION = 1
AST_SCHEMA_VERSION = 1
ADAPTIVE_SCHEMA_VERSION = 1

HELD_OUT_SIZE = 50
HELD_OUT_PER_ATTACK = 25

_STATUSES = frozenset({"completed", "skipped", "failed"})
_SETTINGS = frozenset({"base", "enhanced"})
_EXECUTION_KEYS = ("execution_step_1", "execution_step_2")

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


def parse_models(index: Mapping[str, Any]) -> tuple[BundleModel, ...]:
    entries = require_list(index, "models", BUNDLE_INDEX_NAME)
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
                name=require_text(entry, "name", BUNDLE_INDEX_NAME),
                provider=require_text(entry, "provider", BUNDLE_INDEX_NAME),
                model_id=require_text(entry, "model_id", BUNDLE_INDEX_NAME),
                tier=require_text(entry, "tier", BUNDLE_INDEX_NAME),
                enabled=enabled,
            )
        )
    require_unique((model.name for model in models), label="model name")
    return tuple(models)


def load_splits(root: Path) -> tuple[SplitManifest, ...]:
    directory = root / "splits"
    if not directory.is_dir():
        raise BundleValidationError("splits/: bundle must contain at least one split manifest")
    splits: list[SplitManifest] = []
    for path in sorted(directory.glob("*.json")):
        relative = path.relative_to(root).as_posix()
        payload = load_json(root, relative)
        if require_int(payload, "schema_version", relative) != SPLIT_SCHEMA_VERSION:
            raise BundleValidationError(f"{relative}: unsupported split schema_version")
        setting = require_text(payload, "setting", relative)
        if setting not in _SETTINGS:
            raise BundleValidationError(f"{relative}: setting must be 'base' or 'enhanced'")
        held_out = string_list(payload, "held_out", relative)
        calibration = string_list(payload, "calibration", relative)
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
                seed=require_int(payload, "seed", relative),
                calibration=tuple(calibration),
                held_out=tuple(held_out),
            )
        )
    if not splits:
        raise BundleValidationError("splits/: bundle must contain at least one split manifest")
    return tuple(splits)


# ---------------------------------------------------------------------------
# Native configuration validation
# ---------------------------------------------------------------------------


def validate_configuration(
    root: Path,
    entry: Any,
    models: Sequence[BundleModel],
    splits: Sequence[SplitManifest],
    position: int,
) -> NativeConfiguration:
    label = f"{BUNDLE_INDEX_NAME}: configurations[{position}]"
    if not isinstance(entry, Mapping):
        raise BundleValidationError(f"{label} must be an object")
    configuration_id = require_text(entry, "configuration_id", label)
    label = f"{BUNDLE_INDEX_NAME}: configuration {configuration_id!r}"

    model_name = require_text(entry, "model", label)
    catalog = next((model for model in models if model.name == model_name), None)
    if catalog is None:
        raise BundleValidationError(f"{label} names unknown model {model_name!r}")
    tier = require_text(entry, "tier", label)
    if tier != catalog.tier:
        raise BundleValidationError(
            f"{label} declares tier {tier!r} but the catalog says {catalog.tier!r}"
        )
    setting = require_text(entry, "setting", label)
    if setting not in _SETTINGS:
        raise BundleValidationError(f"{label} setting must be 'base' or 'enhanced'")
    prompt_condition = require_text(entry, "prompt_condition", label)
    split_id = require_text(entry, "split", label)
    split = next((item for item in splits if item.split_id == split_id), None)

    status = require_text(entry, "status", label)
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
    checkpoint_path = require_text(entry, "checkpoint", label)
    run_path = require_text(entry, "run", label)
    manifest_path = require_text(entry, "manifest", label)

    records = load_checkpoint_records(root, checkpoint_path)
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
    require_unique((replay.defense for replay in replays), label=f"{label} replay defense")

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
    envelope = load_json(root, relative)
    if set(envelope) != {"payload", "sha256"}:
        raise BundleValidationError(f"{relative}: expected a {{'payload','sha256'}} envelope")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise BundleValidationError(f"{relative}: manifest payload must be an object")
    expected_ids = string_list(payload, "expected_case_ids", relative)
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
    payload = load_json(root, relative)
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise BundleValidationError(f"{relative}: missing 'protocol' object")
    defense = require_text(protocol, "defense", relative)
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
        validate_record_digest(stripped, label=f"{label}:{case_id}")
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
    payload = load_json(root, relative)
    if require_int(payload, "schema_version", relative) != ENVIRONMENT_SCHEMA_VERSION:
        raise BundleValidationError(f"{relative}: unsupported environment schema_version")
    require_timestamp(payload, "captured_utc", relative)

    platform = payload.get("platform")
    if not isinstance(platform, Mapping):
        raise BundleValidationError(f"{relative}: 'platform' must be an object")
    require_text(platform, "python_version", relative)

    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping) or not isinstance(gpu.get("available"), bool):
        raise BundleValidationError(f"{relative}: 'gpu.available' must be a boolean")
    if gpu["available"]:
        devices = require_list(gpu, "devices", relative)
        if not devices:
            raise BundleValidationError(f"{relative}: 'gpu.devices' must not be empty")
        for device in devices:
            if not isinstance(device, Mapping):
                raise BundleValidationError(f"{relative}: each GPU device must be an object")
            require_text(device, "name", relative)

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
    run_id = require_text(entry, "run_id", BUNDLE_INDEX_NAME)
    label = f"{BUNDLE_INDEX_NAME}: {kind} run {run_id!r}"
    status = require_text(entry, "status", label)
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
    if require_int(entry, "schema_version", label) != expected:
        raise BundleValidationError(
            f"{label}: unsupported record schema_version "
            f"{entry['schema_version']!r}; expected {expected}"
        )


def validate_ast_run(root: Path, entry: Any, models: Sequence[BundleModel]) -> ASTRun:
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

    payload = load_json(root, path)
    records = require_list(payload, "records", path)
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
    if canonical_json(recomputed) != canonical_json(require_mapping(payload, "summary", path)):
        raise BundleValidationError(f"{path}: 'summary' does not recompute from its records")
    return ASTRun(
        run_id=run_id,
        model=str(model),
        status=status,
        status_reason=status_reason,
        records=tuple(dict(record) for record in records),
        summary=recomputed,
    )


def validate_adaptive_run(root: Path, entry: Any) -> AdaptiveRun:
    run_id, status, status_reason, path = _validate_run_entry(entry, kind="adaptive")
    label = f"{BUNDLE_INDEX_NAME}: adaptive run {run_id!r}"
    assert isinstance(entry, Mapping)
    _validate_schema_version(entry, ADAPTIVE_SCHEMA_VERSION, label)
    if status != "completed" or path is None:
        return AdaptiveRun(
            run_id=run_id, status=status, status_reason=status_reason, records=(), summary={}
        )

    payload = load_json(root, path)
    records = require_list(payload, "records", path)
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
    if canonical_json(recomputed) != canonical_json(require_mapping(payload, "summary", path)):
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
    expected_ids = string_list(manifest_payload, "expected_case_ids", label)
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
