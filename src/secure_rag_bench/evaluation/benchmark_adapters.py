"""Optional adapters for external prompt-injection benchmark inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InjecAgentCase:
    """One InjecAgent record normalized without changing its trust boundary."""

    case_id: str
    user_query: str
    untrusted_content: str
    attack_goal: str
    attack_type: str
    user_tool: str
    source_name: str
    attacker_tools: tuple[str, ...]


def load_injecagent_cases(
    path: str | Path, *, source_name: str | None = None
) -> list[InjecAgentCase]:
    """Load official InjecAgent JSON test cases as user query plus tainted tool text."""
    source = Path(path)
    normalized_source = source_name or source.stem
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("InjecAgent test-case file must contain a JSON list")

    cases: list[InjecAgentCase] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"InjecAgent record {index} must be an object")
        cases.append(
            InjecAgentCase(
                case_id=f"injecagent-{index:04d}",
                user_query=_required_text(record, "User Instruction", index),
                untrusted_content=_required_text(record, "Tool Response", index),
                attack_goal=_required_text(record, "Expected Achievements", index),
                attack_type=_required_text(record, "Attack Type", index),
                user_tool=_required_text(record, "User Tool", index),
                source_name=normalized_source,
                attacker_tools=_tool_names(record.get("Attacker Tools"), index),
            )
        )
    return cases


def build_agentdojo_command(
    *, suite: str, model: str, attack: str, logdir: str
) -> list[str]:
    """Build (but never execute) AgentDojo's official benchmark command."""
    return [
        "python", "-m", "agentdojo.scripts.benchmark",
        "--suite", suite,
        "--model", model,
        "--attack", attack,
        "--logdir", logdir,
    ]


def _required_text(record: dict[str, Any], key: str, index: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"InjecAgent record {index} is missing non-empty '{key}'")
    return value


def _tool_names(value: Any, index: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"InjecAgent record {index} has invalid 'Attacker Tools'")
    return tuple(value)
