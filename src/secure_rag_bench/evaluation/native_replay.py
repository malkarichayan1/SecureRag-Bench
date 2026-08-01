"""Offline execution-policy replay for saved native InjecAgent trajectories."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from secure_rag_bench.evaluation.native_monitor import (
    NATIVE_DEFENSES,
    gate_native_action,
)


def replay_task_alignment(
    records: Sequence[Mapping[str, Any]], defense: str
) -> list[dict[str, Any]]:
    """Apply an execution gate to copied official-success trajectories."""
    if defense not in NATIVE_DEFENSES:
        raise ValueError(f"unsupported native defense: {defense}")

    replayed: list[dict[str, Any]] = []
    for source in records:
        record = deepcopy(dict(source))
        if record.get("eval") == "succ":
            record["execution_step_1"] = asdict(
                gate_native_action(
                    proposed_tool=_attacker_tool(record, 0),
                    user_tool=str(record["User Tool"]),
                    defense=defense,
                )
            )
        if record.get("attack") == "ds" and record.get("eval Step 2") == "succ":
            record["execution_step_2"] = asdict(
                gate_native_action(
                    proposed_tool=_attacker_tool(record, 1),
                    user_tool=str(record["User Tool"]),
                    defense=defense,
                )
            )
        replayed.append(record)
    return replayed


def _attacker_tool(record: Mapping[str, Any], index: int) -> str:
    tools = record.get("Attacker Tools")
    if not isinstance(tools, list) or len(tools) <= index:
        case_id = record.get("case_id", "<unknown>")
        raise ValueError(f"case {case_id} is missing attacker tool {index + 1}")
    return str(tools[index])
