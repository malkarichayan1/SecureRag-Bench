"""Stable InjecAgent case loading and held-out pilot selection."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class NativeCase:
    """One native benchmark item with a dataset-stable identifier."""

    attack: str
    item: dict[str, Any]
    case_id: str


def load_native_cases(root: Path, setting: str) -> list[NativeCase]:
    """Load direct-harm and data-stealing cases for one InjecAgent setting."""
    cases: list[NativeCase] = []
    for attack in ("dh", "ds"):
        source = root / "data" / f"test_cases_{attack}_{setting}.json"
        items = json.loads(source.read_text(encoding="utf-8"))
        cases.extend(
            NativeCase(attack=attack, item=item, case_id=_case_id(attack, item))
            for item in items
        )
    return cases


def build_validity_split(
    cases: Sequence[NativeCase],
    calibration_ids: set[str],
    seed: int = 20260801,
) -> dict[str, list[NativeCase]]:
    """Select 25 non-calibration cases per attack, stratified by toolkit."""
    toolkits = _load_toolkits(_benchmark_root() / "data" / "tools.json")
    calibration = [case for case in cases if case.case_id in calibration_ids]
    eligible = [case for case in cases if case.case_id not in calibration_ids]
    held_out: list[NativeCase] = []
    rng = random.Random(seed)

    for attack in ("dh", "ds"):
        attack_cases = [case for case in eligible if case.attack == attack]
        if len(attack_cases) < 25:
            raise ValueError(f"{attack} has fewer than 25 eligible cases")

        by_toolkit: dict[str, list[NativeCase]] = defaultdict(list)
        for case in attack_cases:
            user_tool = case.item["User Tool"]
            by_toolkit[toolkits.get(user_tool, user_tool)].append(case)
        for group in by_toolkit.values():
            group.sort(key=lambda case: case.case_id)
            rng.shuffle(group)

        toolkits_in_attack = sorted(by_toolkit)
        selected_for_attack = 0
        while selected_for_attack < 25:
            for toolkit in toolkits_in_attack:
                if by_toolkit[toolkit]:
                    held_out.append(by_toolkit[toolkit].pop())
                    selected_for_attack += 1
                    if selected_for_attack == 25:
                        break

    return {"calibration": calibration, "held_out": held_out}


def _case_id(attack: str, item: dict[str, Any]) -> str:
    canonical = json.dumps(
        {
            "attack": attack,
            "User Instruction": item["User Instruction"],
            "Tool Response": item["Tool Response"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _benchmark_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "external" / "InjecAgent"


def _load_toolkits(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"{toolkit['toolkit']}{tool['name']}": toolkit["toolkit"]
        for toolkit in data
        for tool in toolkit["tools"]
    }
