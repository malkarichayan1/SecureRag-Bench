"""The parsed data model of a validated study bundle.

A pure leaf module: it holds the frozen dataclasses that
``study_bundle_validation`` produces and that ``study_reporting`` aggregates
into ``ValidatedStudyBundle``. It imports nothing from this package, so it can
be depended on from anywhere in the stack without risking an import cycle.

These types are re-exported by ``study_reporting`` -- import them from there
rather than from here. See ``study_reporting``'s module docstring for the
normative bundle schema each of these mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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
