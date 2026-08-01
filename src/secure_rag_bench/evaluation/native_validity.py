"""Non-scoring explanations for official InjecAgent validity decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ValidityDiagnostic:
    """A project-owned explanation that leaves official parsing untouched."""

    syntax_valid: bool
    protocol_valid: bool
    category: str
    detail: str


def diagnose_native_output(
    raw_output: str,
    official_parse: Mapping[str, Any],
    available_tools: Sequence[str],
    user_tool: str,
) -> ValidityDiagnostic:
    """Explain an official parse result without re-parsing or changing its fields."""
    del raw_output, available_tools, user_tool
    reason = str(official_parse.get("invalid_reason", ""))
    if official_parse.get("eval") != "invalid":
        return ValidityDiagnostic(True, True, "valid", "official parser accepted trajectory")
    category = classify_invalid_reason(reason)
    return ValidityDiagnostic(category != "syntax_error", False, category, reason)


def classify_invalid_reason(reason: str) -> str:
    """Map upstream parser wording to stable project-owned invalid categories."""
    normalized = reason.lower()
    if "recalling of user tool with no observation" in normalized or (
        "recalling of user tool with no next step" in normalized
    ):
        return "user_tool_recall_without_observation"
    if "recalling user tool with fabricated observation" in normalized:
        return "fabricated_observation"
    if "before tool" in normalized:
        return "premature_tool_call"
    if "action not in availabel tools" in normalized:
        return "unavailable_tool"
    return "syntax_error"
