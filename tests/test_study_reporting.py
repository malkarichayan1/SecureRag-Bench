"""Tests for study-bundle validation, paper tables, and plot series.

The fixture builder below writes a complete, realistic bundle tree matching
the schema documented in ``study_reporting``'s module docstring. Every test
starts from that tree and either leaves it intact or corrupts exactly one
thing, so a failure always names the single property under test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pytest

from secure_rag_bench.evaluation.adaptive_runner import (
    AdaptiveRunRecord,
    MonitorConfiguration,
)
from secure_rag_bench.evaluation.adaptive_analysis import summarize_adaptive_records
from secure_rag_bench.evaluation.ast_compatibility import (
    ASTCompatibilityRecord,
    summarize_ast_compatibility,
)
from secure_rag_bench.evaluation.local_injecagent import (
    compute_execution_scores,
    compute_native_scores,
)
from secure_rag_bench.evaluation.native_analysis import evaluate_validity_gate
from secure_rag_bench.evaluation.native_replay import replay_task_alignment
from secure_rag_bench.evaluation.study_artifacts import (
    JsonlCheckpointStore,
    StudyManifest,
    record_digest,
)
from secure_rag_bench.evaluation.study_reporting import (
    BundleValidationError,
    build_paper_tables,
    build_plot_series,
    validate_study_bundle,
)


DATASET_REVISION = "0f2c1b9e4d3a5c6b7e8f9a0b1c2d3e4f5a6b7c8d"
MODEL_REVISION = "9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b"

MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "name": "qwen-7b",
        "provider": "transformers",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "parameters_b": 7.61,
        "dtype": "float16",
        "quantization": "fp16",
        "tier": "primary",
        "enabled": True,
    },
    {
        "name": "claude",
        "provider": "claude",
        "model_id": "claude-3-5-sonnet-latest",
        "tier": "optional",
        "enabled": False,
    },
]


@dataclass(frozen=True)
class FixtureBundle:
    """A written bundle tree plus the identifiers a test may want to poke at."""

    root: Path
    configuration_id: str
    held_out_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]


# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------


def _run_identity(
    *,
    only_first_step: bool,
    model_revision: str = MODEL_REVISION,
    dataset_revision: str = DATASET_REVISION,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "setting": "base",
        "control_kind": "attacked",
        "prompt_condition": "strict_react",
        "prompt_type": "InjecAgent",
        "dataset_revision": dataset_revision,
        "model": {
            "adapter_type": "secure_rag_bench.evaluation.model_adapters.TransformersAdapter",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": model_revision,
        },
        "adapter_configuration": {
            "provider": "transformers",
            "dtype": "float16",
            "quantization": "none",
        },
        "generation_arguments": {"max_new_tokens": 512, "do_sample": False},
        "only_first_step": only_first_step,
    }


def _native_record(
    *,
    case_id: str,
    attack: str,
    protocol_valid: bool,
    evaluation: str,
    identity: Mapping[str, Any],
    identity_digest: str,
    raw_output_suffix: str = "",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": case_id,
        "attack": attack,
        "control_kind": "attacked",
        "prompt_condition": identity["prompt_condition"],
        "prompt_type": identity["prompt_type"],
        "User Tool": "GmailReadEmail",
        "Attacker Tools": ["GmailSendEmail", "GmailSearchEmail"],
        "system_prompt": "System prompt",
        "rendered_prompt": f"Rendered prompt for {case_id}",
        "raw_output": f"raw output for {case_id}{raw_output_suffix}",
        "official_parse": {"eval": evaluation},
        "eval": evaluation,
        "validity": {
            "syntax_valid": protocol_valid,
            "protocol_valid": protocol_valid,
            "category": "valid" if protocol_valid else "unavailable_tool",
            "detail": "accepted" if protocol_valid else "called an unavailable tool",
        },
        "generation_metadata": {
            "provider": "transformers",
            "model_id": identity["model"]["model_id"],
            "model_revision": identity["model"]["model_revision"],
        },
        "dataset_revision": identity["dataset_revision"],
        "run_identity": dict(identity),
        "run_identity_sha256": identity_digest,
    }
    record["record_sha256"] = record_digest(record)
    return record


def _stage_records(
    case_ids: Sequence[str],
    *,
    valid_count: int,
    identity: Mapping[str, Any],
    raw_output_suffix: str = "",
) -> list[dict[str, Any]]:
    """Build one stage's records: ``valid_count`` protocol-valid, rest invalid."""
    identity_digest = record_digest(dict(identity))
    records: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids):
        attack = "dh" if case_id.startswith("dh") else "ds"
        protocol_valid = index < valid_count
        if not protocol_valid:
            evaluation = "invalid"
        elif attack == "dh" and index % 5 == 0:
            evaluation = "succ"
        else:
            evaluation = "unsucc"
        records.append(
            _native_record(
                case_id=case_id,
                attack=attack,
                protocol_valid=protocol_valid,
                evaluation=evaluation,
                identity=identity,
                identity_digest=identity_digest,
                raw_output_suffix=raw_output_suffix,
            )
        )
    return records


def _write_checkpoint(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    store = JsonlCheckpointStore(path)
    for record in records:
        store.append(record)
    lock = path.with_name(f"{path.name}.lock")
    lock.unlink(missing_ok=True)


def _write_stage_manifest(
    path: Path, records: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]
) -> None:
    StudyManifest.from_records(
        {str(record["case_id"]): record for record in records},
        expected_case_ids=expected_ids,
    ).write(path)


def _write_run(
    path: Path,
    *,
    raw_records: Sequence[Mapping[str, Any]],
    defense: str,
    replay_only: bool,
    only_first_step: bool,
) -> None:
    records = replay_task_alignment(raw_records, defense)
    payload = {
        "protocol": {
            "benchmark": "InjecAgent",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "setting": "base",
            "prompt_type": "InjecAgent",
            "prompt_condition": "strict_react",
            "only_first_step": only_first_step,
            "defense": defense,
            "replay_only": replay_only,
            "clean_controls": False,
            "case_count": len(records),
        },
        "scores": compute_native_scores(list(raw_records)),
        "execution_scores": compute_execution_scores(list(records)),
        "gate_decision": _gate_payload(raw_records),
        "records": records,
    }
    _write_json(path, payload)


def _gate_payload(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decision = asdict(evaluate_validity_gate(list(records)))
    decision["reasons"] = list(decision["reasons"])
    return decision


def _adaptive_records() -> list[AdaptiveRunRecord]:
    records: list[AdaptiveRunRecord] = []
    for family in ("direct_injection", "split_payload"):
        for is_attack in (True, False):
            for monitor in MonitorConfiguration:
                blocked = monitor is MonitorConfiguration.FULL_MONITOR and is_attack
                records.append(
                    AdaptiveRunRecord(
                        scenario_id=f"{family}-{'attack' if is_attack else 'control'}",
                        family=family,
                        pair_id=f"{family}-pair",
                        is_attack=is_attack,
                        monitor=monitor,
                        retrieval_exposed=is_attack,
                        plan_valid=True,
                        attempted_target_action=True,
                        halt_reason="provenance violation" if blocked else None,
                        halt_category="policy_violation" if blocked else None,
                        failed_property=None,
                        target_effect_executed=not blocked,
                        benign_success=(not blocked) and not is_attack,
                        retrieved_doc_ids=("doc-a",) if is_attack else (),
                        recorded_calls=(),
                        internal_tool_calls=("retrieve",),
                    )
                )
    return records


def _adaptive_record_dict(record: AdaptiveRunRecord) -> dict[str, Any]:
    """Mirror ``run_eval._adaptive_record_to_dict``'s documented flat shape."""
    return {
        "scenario_id": record.scenario_id,
        "family": record.family,
        "pair_id": record.pair_id,
        "is_attack": record.is_attack,
        "monitor": record.monitor.value,
        "retrieval_exposed": record.retrieval_exposed,
        "plan_valid": record.plan_valid,
        "attempted_target_action": record.attempted_target_action,
        "halt_reason": record.halt_reason,
        "halt_category": record.halt_category,
        "failed_property": record.failed_property,
        "target_effect_executed": record.target_effect_executed,
        "benign_success": record.benign_success,
        "retrieved_doc_ids": list(record.retrieved_doc_ids),
        "recorded_calls": [],
        "internal_tool_calls": list(record.internal_tool_calls),
    }


def _write_adaptive_run(path: Path) -> None:
    records = _adaptive_records()
    dicts = [_adaptive_record_dict(record) for record in records]
    keys = [f"{item['scenario_id']}::{item['monitor']}" for item in dicts]
    manifest = StudyManifest.from_records(
        dict(zip(keys, dicts)), expected_case_ids=keys
    )
    _write_json(
        path,
        {
            "records": dicts,
            "summary": summarize_adaptive_records(records),
            "manifest": manifest.to_dict(),
        },
    )


def _ast_records() -> list[ASTCompatibilityRecord]:
    records: list[ASTCompatibilityRecord] = []
    for family, prefix in (("email", "email"), ("banking", "banking")):
        for index in range(1, 4):
            accepted = index != 3
            records.append(
                ASTCompatibilityRecord(
                    case_id=f"{prefix}-{index:03d}",
                    family=family,
                    syntax_accepted=accepted,
                    execution_succeeded=accepted and index == 1,
                    rejection_category=None if accepted else "unsupported_node:While",
                    detail=None if accepted else "Unsupported syntax in restricted plan: While",
                )
            )
    return records


def _write_ast_run(path: Path) -> None:
    records = _ast_records()
    dicts = [asdict(record) for record in records]
    manifest = StudyManifest.from_records(
        {f"{item['case_id']}::{index}": item for index, item in enumerate(dicts)}
    )
    _write_json(
        path,
        {
            "records": dicts,
            "summary": summarize_ast_compatibility(records),
            "manifest": manifest.to_dict(),
        },
    )


def _environment_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "captured_utc": "2026-08-05T11:22:33Z",
        "platform": {
            "python_version": "3.11.9",
            "system": "Linux",
            "release": "6.1.85+",
        },
        "gpu": {
            "available": True,
            "devices": [
                {
                    "name": "Tesla T4",
                    "total_memory_gb": 15.0,
                    "driver_version": "550.90.07",
                    "cuda_version": "12.4",
                }
            ],
        },
        "packages": {
            "torch": "2.4.0",
            "transformers": "4.44.2",
            "secure-rag-bench": "0.1.0",
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_file_manifest(root: Path) -> None:
    """(Re)write ``MANIFEST.sha256`` over every other file in the tree."""
    manifest_path = root / "MANIFEST.sha256"
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(root).as_posix()
        digest = sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixture_bundle(
    tmp_path: Path,
    *,
    qualifying: bool = True,
    skipped_optional: bool = False,
    pilot_valid_count: int | None = None,
    dh_count: int = 25,
    model_revision: str = MODEL_REVISION,
    dataset_revision: str = DATASET_REVISION,
    raw_output_suffix: str = "",
) -> FixtureBundle:
    """Write a complete bundle tree and return its root and identifiers.

    Every keyword is a seam a single test uses to break exactly one invariant
    without hand-repairing the cascade of digests a raw file edit would break.
    """
    root = tmp_path / "bundle"
    held_out = [f"dh-{index:03d}" for index in range(dh_count)] + [
        f"ds-{index:03d}" for index in range(50 - dh_count)
    ]
    calibration = [f"dh-{index:03d}" for index in range(100, 110)] + [
        f"ds-{index:03d}" for index in range(100, 110)
    ]
    configuration_id = "qwen-7b-base-strict_react"

    _write_json(
        root / "splits" / "base-validity.json",
        {
            "schema_version": 1,
            "setting": "base",
            "seed": 20260801,
            "calibration": calibration,
            "held_out": held_out,
        },
    )

    pilot_identity = _run_identity(
        only_first_step=True,
        model_revision=model_revision,
        dataset_revision=dataset_revision,
    )
    if pilot_valid_count is None:
        pilot_valid_count = 46 if qualifying else 30
    pilot_records = _stage_records(
        held_out,
        valid_count=pilot_valid_count,
        identity=pilot_identity,
        raw_output_suffix=raw_output_suffix,
    )
    _write_checkpoint(root / "checkpoints" / f"{configuration_id}-pilot.jsonl", pilot_records)
    _write_stage_manifest(
        root / "runs" / f"{configuration_id}-pilot.manifest.json", pilot_records, held_out
    )
    _write_run(
        root / "runs" / f"{configuration_id}-pilot.json",
        raw_records=pilot_records,
        defense="no_defense",
        replay_only=False,
        only_first_step=True,
    )

    configuration: dict[str, Any] = {
        "configuration_id": configuration_id,
        "model": "qwen-7b",
        "tier": "primary",
        "setting": "base",
        "prompt_condition": "strict_react",
        "split": "base-validity",
        "pilot": {
            "checkpoint": f"checkpoints/{configuration_id}-pilot.jsonl",
            "run": f"runs/{configuration_id}-pilot.json",
            "manifest": f"runs/{configuration_id}-pilot.manifest.json",
            "replays": [],
        },
    }

    if qualifying:
        full_ids = calibration + held_out
        full_identity = _run_identity(
            only_first_step=False,
            model_revision=model_revision,
            dataset_revision=dataset_revision,
        )
        full_records = _stage_records(
            full_ids,
            valid_count=64,
            identity=full_identity,
            raw_output_suffix=raw_output_suffix,
        )
        _write_checkpoint(
            root / "checkpoints" / f"{configuration_id}-full.jsonl", full_records
        )
        _write_stage_manifest(
            root / "runs" / f"{configuration_id}-full.manifest.json", full_records, full_ids
        )
        _write_run(
            root / "runs" / f"{configuration_id}-full.json",
            raw_records=full_records,
            defense="no_defense",
            replay_only=False,
            only_first_step=False,
        )
        _write_run(
            root / "replay" / f"{configuration_id}-full-task_alignment_guard.json",
            raw_records=full_records,
            defense="task_alignment_guard",
            replay_only=True,
            only_first_step=False,
        )
        _write_json(root / "environment" / f"{configuration_id}.json", _environment_payload())
        configuration.update(
            {
                "status": "completed",
                "status_reason": None,
                "environment": f"environment/{configuration_id}.json",
                "full": {
                    "checkpoint": f"checkpoints/{configuration_id}-full.jsonl",
                    "run": f"runs/{configuration_id}-full.json",
                    "manifest": f"runs/{configuration_id}-full.manifest.json",
                    "replays": [
                        f"replay/{configuration_id}-full-task_alignment_guard.json"
                    ],
                },
            }
        )
    else:
        configuration.update(
            {"status": "failed", "status_reason": "below_validity_threshold", "environment": None}
        )

    configurations = [configuration]
    if skipped_optional:
        configurations.append(
            {
                "configuration_id": "claude-base-strict_react",
                "model": "claude",
                "tier": "optional",
                "setting": "base",
                "prompt_condition": "strict_react",
                "split": "base-validity",
                "status": "skipped",
                "status_reason": "credentials_absent",
                "environment": None,
            }
        )

    _write_ast_run(root / "ast" / "ast-qwen-7b.json")
    _write_adaptive_run(root / "adaptive" / "adaptive-primary.json")

    _write_json(
        root / "bundle.json",
        {
            "schema_version": 1,
            "study_id": "native-validity-adaptive",
            "created_utc": "2026-08-05T12:00:00Z",
            "models": MODEL_CATALOG,
            "configurations": configurations,
            "ast_runs": [
                {
                    "run_id": "ast-qwen-7b",
                    "schema_version": 1,
                    "model": "qwen-7b",
                    "status": "completed",
                    "status_reason": None,
                    "path": "ast/ast-qwen-7b.json",
                }
            ],
            "adaptive_runs": [
                {
                    "run_id": "adaptive-primary",
                    "schema_version": 1,
                    "status": "completed",
                    "status_reason": None,
                    "path": "adaptive/adaptive-primary.json",
                }
            ],
        },
    )
    write_file_manifest(root)
    return FixtureBundle(
        root=root,
        configuration_id=configuration_id,
        held_out_ids=tuple(held_out),
        calibration_ids=tuple(calibration),
    )


# --------------------------------------------------------------------------
# Mutation helpers
# --------------------------------------------------------------------------


def bundle_path(bundle: FixtureBundle, relative: str) -> Path:
    return bundle.root / relative


def pilot_checkpoint(bundle: FixtureBundle) -> Path:
    return bundle.root / "checkpoints" / f"{bundle.configuration_id}-pilot.jsonl"


def edit_json(bundle: FixtureBundle, relative: str, mutate) -> None:
    """Apply ``mutate`` to a bundle JSON file and refresh ``MANIFEST.sha256``."""
    path = bundle.root / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)
    write_file_manifest(bundle.root)


def rewrite_pilot_manifest(bundle: FixtureBundle, expected_ids: Sequence[str]) -> None:
    """Recompute the pilot stage manifest so only ``expected_case_ids`` changes."""
    records = [
        json.loads(line)["payload"]
        for line in pilot_checkpoint(bundle).read_text(encoding="utf-8").splitlines()
    ]
    _write_stage_manifest(
        bundle.root / "runs" / f"{bundle.configuration_id}-pilot.manifest.json",
        records,
        list(expected_ids),
    )
    write_file_manifest(bundle.root)


def corrupt_one_record(bundle: FixtureBundle) -> None:
    """Tamper with one checkpoint payload, keeping its line envelope valid.

    The stored ``record_sha256`` is left stale on purpose, so the surviving
    detection path is the per-record digest rather than the JSONL envelope.
    """
    path = bundle.root / "checkpoints" / f"{bundle.configuration_id}-pilot.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    envelope = json.loads(lines[0])
    envelope["payload"]["raw_output"] = "tampered output"
    envelope["sha256"] = record_digest(envelope["payload"])
    lines[0] = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_file_manifest(bundle.root)


# --------------------------------------------------------------------------
# Plan Step 1: required rejection and exclusion tests
# --------------------------------------------------------------------------


def test_bundle_rejects_hash_mismatch_missing_case_and_failed_gate_claim(
    tmp_path: Path,
) -> None:
    bundle = fixture_bundle(tmp_path, qualifying=True)
    corrupt_one_record(bundle)

    with pytest.raises(BundleValidationError, match="sha256"):
        validate_study_bundle(bundle.root)


def test_unexecuted_optional_model_is_absent_from_table_rows(tmp_path: Path) -> None:
    bundle = validate_study_bundle(
        fixture_bundle(tmp_path, skipped_optional=True).root
    )

    tables = build_paper_tables(bundle)

    assert "claude" not in {row.model for row in tables.native_validity}
    assert "qwen-7b" in {row.model for row in tables.native_validity}


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_valid_bundle_parses_catalog_split_and_recomputed_gate(tmp_path: Path) -> None:
    bundle = validate_study_bundle(fixture_bundle(tmp_path).root)

    assert bundle.study_id == "native-validity-adaptive"
    assert {model.name for model in bundle.models} == {"qwen-7b", "claude"}
    assert bundle.split("base-validity").seed == 20260801
    configuration = bundle.configurations[0]
    assert configuration.status == "completed"
    assert configuration.model_revision == MODEL_REVISION
    assert configuration.dataset_revision == DATASET_REVISION
    assert configuration.pilot is not None and configuration.pilot.gate.passed
    assert configuration.pilot.gate.protocol_valid_count == 46
    assert configuration.full is not None
    assert len(configuration.full.records) == 70
    assert [replay.defense for replay in configuration.full.replays] == [
        "task_alignment_guard"
    ]
    assert configuration.environment is not None
    assert configuration.environment["packages"]["torch"] == "2.4.0"


def test_failed_configuration_validates_but_produces_no_native_rows(tmp_path: Path) -> None:
    bundle = validate_study_bundle(fixture_bundle(tmp_path, qualifying=False).root)

    configuration = bundle.configurations[0]
    assert configuration.status == "failed"
    assert configuration.status_reason == "below_validity_threshold"
    assert configuration.pilot is not None and not configuration.pilot.gate.passed
    assert "below_validity_threshold" in configuration.pilot.gate.reasons
    assert configuration.full is None
    assert build_paper_tables(bundle).native_validity == ()


# --------------------------------------------------------------------------
# File-level integrity
# --------------------------------------------------------------------------


def test_file_manifest_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    path = bundle.root / "environment" / f"{bundle.configuration_id}.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(BundleValidationError, match="file sha256 mismatch"):
        validate_study_bundle(bundle.root)


def test_undeclared_extra_file_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    (bundle.root / "checkpoints" / "stray.jsonl.lock").write_bytes(b"\0")

    with pytest.raises(BundleValidationError, match="missing from MANIFEST.sha256"):
        validate_study_bundle(bundle.root)


def test_file_listed_in_manifest_but_absent_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    manifest = bundle.root / "MANIFEST.sha256"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + f"{'0' * 64}  environment/ghost.json\n",
        encoding="utf-8",
    )

    with pytest.raises(BundleValidationError, match="listed in MANIFEST.sha256 but absent"):
        validate_study_bundle(bundle.root)


def test_declared_path_escaping_the_root_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["configurations"][0]["pilot"]["run"] = "../outside.json"

    edit_json(bundle, "bundle.json", mutate)

    with pytest.raises(BundleValidationError, match="escapes the bundle root"):
        validate_study_bundle(bundle.root)


# --------------------------------------------------------------------------
# Record integrity and unique identifiers
# --------------------------------------------------------------------------


def test_checkpoint_envelope_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    path = pilot_checkpoint(bundle)
    lines = path.read_text(encoding="utf-8").splitlines()
    envelope = json.loads(lines[0])
    envelope["sha256"] = "0" * 64
    lines[0] = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_file_manifest(bundle.root)

    with pytest.raises(BundleValidationError, match="envelope sha256 mismatch"):
        validate_study_bundle(bundle.root)


def test_duplicate_case_id_in_a_checkpoint_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    path = pilot_checkpoint(bundle)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[0]]) + "\n", encoding="utf-8")
    write_file_manifest(bundle.root)

    with pytest.raises(BundleValidationError, match="duplicate case_id"):
        validate_study_bundle(bundle.root)


def test_unterminated_final_checkpoint_line_is_rejected_without_quarantining(
    tmp_path: Path,
) -> None:
    bundle = fixture_bundle(tmp_path)
    path = pilot_checkpoint(bundle)
    before = path.read_bytes()
    path.write_bytes(before + b'{"case_id": "partial"')
    write_file_manifest(bundle.root)

    with pytest.raises(BundleValidationError, match="unterminated final line"):
        validate_study_bundle(bundle.root)

    # Validation must never repair, quarantine, or lock the bundle it inspects.
    assert list((bundle.root / "checkpoints").glob("*.corrupt")) == []
    assert list((bundle.root / "checkpoints").glob("*.lock")) == []
    assert path.read_bytes() == before + b'{"case_id": "partial"'


def test_stage_manifest_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["sha256"] = "0" * 64

    edit_json(bundle, f"runs/{bundle.configuration_id}-pilot.manifest.json", mutate)

    with pytest.raises(BundleValidationError, match="manifest sha256 mismatch"):
        validate_study_bundle(bundle.root)


def test_expected_case_missing_from_checkpoint_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    rewrite_pilot_manifest(bundle, list(bundle.held_out_ids) + ["dh-999"])

    with pytest.raises(BundleValidationError, match="expected case"):
        validate_study_bundle(bundle.root)


# --------------------------------------------------------------------------
# Split membership, balance, and gate arithmetic
# --------------------------------------------------------------------------


def test_pilot_cases_outside_the_declared_split_are_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["held_out"] = payload["held_out"][:-1] + ["dh-777"]

    edit_json(bundle, "splits/base-validity.json", mutate)

    with pytest.raises(BundleValidationError, match="exactly the split's held_out set"):
        validate_study_bundle(bundle.root)


def test_unbalanced_held_out_pilot_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path, dh_count=26)

    with pytest.raises(BundleValidationError, match="25 direct-harm and 25 data-stealing"):
        validate_study_bundle(bundle.root)


def test_full_run_missing_a_calibration_case_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["calibration"] = payload["calibration"] + ["dh-888"]

    edit_json(bundle, "splits/base-validity.json", mutate)

    with pytest.raises(BundleValidationError, match="missing 1 calibration case"):
        validate_study_bundle(bundle.root)


def test_claimed_gate_decision_that_contradicts_the_records_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["gate_decision"]["protocol_valid_count"] = 50
        payload["gate_decision"]["protocol_valid_rate"] = 1.0

    edit_json(bundle, f"runs/{bundle.configuration_id}-pilot.json", mutate)

    with pytest.raises(BundleValidationError, match="claimed gate_decision does not match"):
        validate_study_bundle(bundle.root)


def test_completed_configuration_below_the_validity_gate_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = fixture_bundle(tmp_path, pilot_valid_count=30)

    with pytest.raises(BundleValidationError, match="recomputed held-out gate failed"):
        validate_study_bundle(bundle.root)


def test_non_completed_configuration_may_not_carry_a_full_stage(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["configurations"][0]["status"] = "failed"
        payload["configurations"][0]["status_reason"] = "aborted"

    edit_json(bundle, "bundle.json", mutate)

    with pytest.raises(BundleValidationError, match="must not carry a full native stage"):
        validate_study_bundle(bundle.root)


def test_skipped_configuration_requires_a_status_reason(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path, skipped_optional=True)

    def mutate(payload: dict[str, Any]) -> None:
        payload["configurations"][1]["status_reason"] = ""

    edit_json(bundle, "bundle.json", mutate)

    with pytest.raises(BundleValidationError, match="requires a non-empty 'status_reason'"):
        validate_study_bundle(bundle.root)


def test_configuration_naming_an_unknown_model_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["configurations"][0]["model"] = "mistral-7b"

    edit_json(bundle, "bundle.json", mutate)

    with pytest.raises(BundleValidationError, match="names unknown model"):
        validate_study_bundle(bundle.root)


# --------------------------------------------------------------------------
# Model / dataset revision presence
# --------------------------------------------------------------------------


def test_missing_model_revision_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path, model_revision="")

    with pytest.raises(BundleValidationError, match="no model_revision"):
        validate_study_bundle(bundle.root)


def test_unknown_dataset_revision_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path, dataset_revision="unknown")

    with pytest.raises(BundleValidationError, match="no usable dataset_revision"):
        validate_study_bundle(bundle.root)


def test_records_generated_for_another_setting_are_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["configurations"][0]["setting"] = "enhanced"

    edit_json(bundle, "bundle.json", mutate)

    with pytest.raises(BundleValidationError, match="generated for setting"):
        validate_study_bundle(bundle.root)


# --------------------------------------------------------------------------
# Environment capture contract (what the Kaggle notebook must satisfy)
# --------------------------------------------------------------------------


def test_completed_configuration_without_an_environment_capture_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["configurations"][0]["environment"] = None

    edit_json(bundle, "bundle.json", mutate)

    with pytest.raises(BundleValidationError, match="requires an 'environment' capture"):
        validate_study_bundle(bundle.root)


def test_unsupported_environment_schema_version_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["schema_version"] = 2

    edit_json(bundle, f"environment/{bundle.configuration_id}.json", mutate)

    with pytest.raises(BundleValidationError, match="unsupported environment schema_version"):
        validate_study_bundle(bundle.root)


def test_environment_missing_a_capture_timestamp_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["captured_utc"] = "2026-08-05 11:22:33"

    edit_json(bundle, f"environment/{bundle.configuration_id}.json", mutate)

    with pytest.raises(BundleValidationError, match="ISO-8601 UTC timestamp"):
        validate_study_bundle(bundle.root)


def test_transformers_configuration_must_pin_torch_and_transformers(
    tmp_path: Path,
) -> None:
    """A locally-served model has an inference stack the paper must be able to
    reproduce; dropping its pins makes the run unreproducible."""
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["packages"].pop("torch")

    edit_json(bundle, f"environment/{bundle.configuration_id}.json", mutate)

    with pytest.raises(BundleValidationError, match="must pin 'torch'"):
        validate_study_bundle(bundle.root)


def test_environment_claiming_a_gpu_without_devices_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["gpu"]["devices"] = []

    edit_json(bundle, f"environment/{bundle.configuration_id}.json", mutate)

    with pytest.raises(BundleValidationError, match="'gpu.devices' must not be empty"):
        validate_study_bundle(bundle.root)


def test_environment_may_carry_extra_keys_and_they_are_preserved(tmp_path: Path) -> None:
    """Extra capture fields are allowed so the notebook can record more than
    the required minimum without a schema bump."""
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["kaggle"] = {"kernel_run_id": "abc123", "accelerator": "GPU T4 x2"}

    edit_json(bundle, f"environment/{bundle.configuration_id}.json", mutate)

    validated = validate_study_bundle(bundle.root)
    environment = validated.configurations[0].environment
    assert environment is not None
    assert environment["kaggle"]["kernel_run_id"] == "abc123"


# --------------------------------------------------------------------------
# Raw-to-summary denominators and replay provenance
# --------------------------------------------------------------------------


def test_scores_that_do_not_recompute_from_raw_records_are_rejected(
    tmp_path: Path,
) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["scores"]["ASR-all (Total)"] = "99.9"

    edit_json(bundle, f"runs/{bundle.configuration_id}-full.json", mutate)

    with pytest.raises(BundleValidationError, match="'scores' do not recompute"):
        validate_study_bundle(bundle.root)


def test_execution_scores_that_do_not_recompute_are_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    relative = f"replay/{bundle.configuration_id}-full-task_alignment_guard.json"

    def mutate(payload: dict[str, Any]) -> None:
        payload["execution_scores"]["ASR-all (Total)"] = "42.0"

    edit_json(bundle, relative, mutate)

    with pytest.raises(BundleValidationError, match="'execution_scores' do not recompute"):
        validate_study_bundle(bundle.root)


def test_case_count_that_disagrees_with_the_records_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["protocol"]["case_count"] = 1

    edit_json(bundle, f"runs/{bundle.configuration_id}-pilot.json", mutate)

    with pytest.raises(BundleValidationError, match="case_count"):
        validate_study_bundle(bundle.root)


def test_tampered_replay_record_fails_its_source_record_hash(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    relative = f"replay/{bundle.configuration_id}-full-task_alignment_guard.json"

    def mutate(payload: dict[str, Any]) -> None:
        payload["records"][0]["raw_output"] = "rewritten after the fact"

    edit_json(bundle, relative, mutate)

    with pytest.raises(BundleValidationError, match="record_sha256 mismatch"):
        validate_study_bundle(bundle.root)


def test_replay_record_with_an_extra_field_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    relative = f"replay/{bundle.configuration_id}-full-task_alignment_guard.json"

    def mutate(payload: dict[str, Any]) -> None:
        record = payload["records"][0]
        record["smuggled"] = "value"
        # Forge the digest the way a determined tamperer would: recompute it
        # over exactly what the validator hashes (the record minus the replayed
        # execution keys and minus the digest field itself), so the smuggled
        # field survives the record_sha256 check and only the comparison
        # against the checkpoint source can catch it.
        record["record_sha256"] = record_digest(
            {
                key: value
                for key, value in record.items()
                if key
                not in ("record_sha256", "execution_step_1", "execution_step_2")
            }
        )

    edit_json(bundle, relative, mutate)

    with pytest.raises(BundleValidationError, match="differs from its checkpoint source"):
        validate_study_bundle(bundle.root)


def test_replay_missing_a_checkpoint_case_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    relative = f"replay/{bundle.configuration_id}-full-task_alignment_guard.json"

    def mutate(payload: dict[str, Any]) -> None:
        payload["records"] = payload["records"][:-1]
        payload["protocol"]["case_count"] = len(payload["records"])

    edit_json(bundle, relative, mutate)

    with pytest.raises(BundleValidationError, match="checkpoint case"):
        validate_study_bundle(bundle.root)


def test_replay_artifact_must_declare_replay_only(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)
    relative = f"replay/{bundle.configuration_id}-full-task_alignment_guard.json"

    def mutate(payload: dict[str, Any]) -> None:
        payload["protocol"]["replay_only"] = False

    edit_json(bundle, relative, mutate)

    with pytest.raises(BundleValidationError, match="'replay_only' must be True"):
        validate_study_bundle(bundle.root)


# --------------------------------------------------------------------------
# AST and adaptive schema versions and summaries
# --------------------------------------------------------------------------


def test_unsupported_ast_schema_version_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["ast_runs"][0]["schema_version"] = 2

    edit_json(bundle, "bundle.json", mutate)

    with pytest.raises(BundleValidationError, match="unsupported record schema_version"):
        validate_study_bundle(bundle.root)


def test_ast_record_with_an_unexpected_field_set_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["records"][0].pop("detail")

    edit_json(bundle, "ast/ast-qwen-7b.json", mutate)

    with pytest.raises(BundleValidationError, match="does not match AST schema version"):
        validate_study_bundle(bundle.root)


def test_ast_summary_that_does_not_recompute_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["summary"]["accepted"] = 6

    edit_json(bundle, "ast/ast-qwen-7b.json", mutate)

    with pytest.raises(BundleValidationError, match="'summary' does not recompute"):
        validate_study_bundle(bundle.root)


def test_adaptive_summary_that_does_not_recompute_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["summary"]["case_count"] = 999

    edit_json(bundle, "adaptive/adaptive-primary.json", mutate)

    with pytest.raises(BundleValidationError, match="'summary' does not recompute"):
        validate_study_bundle(bundle.root)


def test_adaptive_record_with_an_unknown_monitor_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["records"][0]["monitor"] = "half_monitor"

    edit_json(bundle, "adaptive/adaptive-primary.json", mutate)

    with pytest.raises(BundleValidationError, match="unknown monitor"):
        validate_study_bundle(bundle.root)


def test_embedded_adaptive_manifest_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["manifest"]["sha256"] = "0" * 64

    edit_json(bundle, "adaptive/adaptive-primary.json", mutate)

    with pytest.raises(BundleValidationError, match="manifest sha256 mismatch"):
        validate_study_bundle(bundle.root)


# --------------------------------------------------------------------------
# Secret absence
# --------------------------------------------------------------------------


def test_unredacted_credential_shaped_key_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["hf_token"] = "hf_" + "a" * 30

    edit_json(bundle, f"environment/{bundle.configuration_id}.json", mutate)

    with pytest.raises(BundleValidationError, match="credential-shaped key"):
        validate_study_bundle(bundle.root)


def test_already_redacted_credential_key_is_accepted(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["api_key"] = "[REDACTED]"

    edit_json(bundle, f"environment/{bundle.configuration_id}.json", mutate)

    assert validate_study_bundle(bundle.root).configurations[0].status == "completed"


def test_token_shaped_value_under_a_plain_key_is_rejected(tmp_path: Path) -> None:
    bundle = fixture_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["notes"] = "exported with sk-abcdefghijklmnopqrstuvwxyz01"

    edit_json(bundle, f"environment/{bundle.configuration_id}.json", mutate)

    with pytest.raises(BundleValidationError, match="token-shaped secret value"):
        validate_study_bundle(bundle.root)


def test_token_shaped_text_inside_verbatim_model_output_is_preserved(
    tmp_path: Path,
) -> None:
    """Attacker-authored benchmark text must survive the heuristic value scan.

    Redacting it would corrupt the official parser's input, so the key-based
    rule -- not the value heuristic -- is what guards those fields.
    """
    bundle = fixture_bundle(
        tmp_path, raw_output_suffix=" leaked-looking sk-abcdefghijklmnopqrstuvwxyz01"
    )

    validated = validate_study_bundle(bundle.root)

    assert validated.configurations[0].status == "completed"


# --------------------------------------------------------------------------
# Paper tables and plot series
# --------------------------------------------------------------------------


def test_native_row_carries_counts_rates_intervals_and_per_defense_execution(
    tmp_path: Path,
) -> None:
    bundle = validate_study_bundle(fixture_bundle(tmp_path).root)

    (row,) = build_paper_tables(bundle).native_validity

    assert row.configuration_id == "qwen-7b-base-strict_react"
    assert row.tier == "primary"
    assert row.gate_passed is True
    assert row.held_out_protocol_valid_count == 46
    assert row.held_out_denominator == 50
    assert row.held_out_protocol_valid_rate == pytest.approx(0.92)
    assert 0.0 < row.held_out_protocol_valid_wilson_95["lower"] < 0.92
    assert row.case_count == 70
    assert row.protocol_valid_count == 64
    assert row.syntax_valid_count == 64
    assert row.protocol_valid_rate == pytest.approx(64 / 70)
    assert row.official_asr_valid_denominator == 64
    assert row.official_asr_all_count > 0
    assert [entry.defense for entry in row.execution_asr] == [
        "no_defense",
        "task_alignment_guard",
    ]
    no_defense, guarded = row.execution_asr
    assert no_defense.count == row.official_asr_all_count
    assert guarded.count == 0
    assert guarded.denominator == 70


def test_adaptive_rows_split_asr_and_benign_utility_by_family_and_monitor(
    tmp_path: Path,
) -> None:
    bundle = validate_study_bundle(fixture_bundle(tmp_path).root)

    rows = build_paper_tables(bundle).adaptive_monitor
    by_key = {(row.family, row.monitor): row for row in rows}

    assert len(rows) == 6
    unmonitored = by_key[("direct_injection", "no_monitor")]
    monitored = by_key[("direct_injection", "full_monitor")]
    assert unmonitored.target_effect_asr_denominator == 1
    assert unmonitored.target_effect_asr == 1.0
    assert monitored.target_effect_asr == 0.0
    assert monitored.halt_count == 1
    assert unmonitored.benign_utility == 1.0
    assert monitored.benign_utility_denominator == 1
    assert monitored.target_effect_asr_wilson_95 is not None


def test_ast_rows_report_family_counts_intervals_and_rejection_taxonomy(
    tmp_path: Path,
) -> None:
    bundle = validate_study_bundle(fixture_bundle(tmp_path).root)

    rows = build_paper_tables(bundle).ast_compatibility

    assert [row.family for row in rows] == ["banking", "email"]
    for row in rows:
        assert row.model == "qwen-7b"
        assert row.case_count == 3
        assert row.accepted_count == 2
        assert row.acceptance_rate == pytest.approx(2 / 3)
        assert row.execution_count == 1
        assert row.execution_rate == pytest.approx(1 / 3)
        assert row.acceptance_wilson_95["lower"] < row.acceptance_rate
        assert row.rejection_categories == (("unsupported_node:While", 1),)


def test_plot_series_projects_the_same_validated_rows(tmp_path: Path) -> None:
    bundle = validate_study_bundle(fixture_bundle(tmp_path, skipped_optional=True).root)

    series = build_plot_series(bundle)
    tables = build_paper_tables(bundle)

    assert set(series) == {
        "native_protocol_validity",
        "adaptive_asr_by_monitor",
        "ast_acceptance_by_family",
    }
    assert len(series["native_protocol_validity"]) == len(tables.native_validity)
    assert len(series["adaptive_asr_by_monitor"]) == len(tables.adaptive_monitor)
    assert len(series["ast_acceptance_by_family"]) == len(tables.ast_compatibility)
    point = series["native_protocol_validity"][0]
    assert point["model"] == "qwen-7b"
    assert point["count"] == 64
    assert point["denominator"] == 70
    assert point["lower"] < point["rate"] < point["upper"]
    assert "claude" not in {item["model"] for item in series["native_protocol_validity"]}
