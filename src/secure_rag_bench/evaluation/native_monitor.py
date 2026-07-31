"""Execution gates for native InjecAgent measurements.

Native InjecAgent measures whether a model selects an attacker-controlled tool.
The gate below intentionally reports task-alignment enforcement separately from
the model's original benchmark score; it does not claim to be a provenance
ablation.
"""

from __future__ import annotations

from dataclasses import dataclass


NATIVE_DEFENSES = frozenset({"no_defense", "task_alignment_guard"})


@dataclass(frozen=True)
class NativeExecutionDecision:
    """Whether a model-proposed native benchmark action may execute."""

    allowed: bool
    reason: str
    failed_property: str | None = None


def gate_native_action(
    *,
    proposed_tool: str,
    user_tool: str,
    defense: str,
) -> NativeExecutionDecision:
    """Apply the configured execution gate to one proposed tool action."""
    if defense not in NATIVE_DEFENSES:
        raise ValueError(f"unsupported native defense: {defense}")
    if defense == "no_defense" or proposed_tool == user_tool:
        return NativeExecutionDecision(allowed=True, reason="Execution allowed")
    return NativeExecutionDecision(
        allowed=False,
        reason="Proposed tool is not aligned with the user's requested tool.",
        failed_property="task_alignment",
    )
