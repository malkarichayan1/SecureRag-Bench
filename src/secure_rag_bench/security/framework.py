"""Contextual security framework — four security properties."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from secure_rag_bench.camel.provenance import Capability, Provenance, Source


class SecurityProperty(str, Enum):
    """The four contextual security properties."""

    TASK_ALIGNMENT = "task_alignment"
    ACTION_ALIGNMENT = "action_alignment"
    SOURCE_AUTHORIZATION = "source_authorization"
    DATA_ISOLATION = "data_isolation"


@dataclass
class SecurityContext:
    """Context for evaluating all four security properties."""

    task_description: str
    requested_action: str
    data_provenance: Provenance
    authorized_sources: frozenset[Source] = frozenset({Source.USER})
    allowed_actions: frozenset[str] | None = None
    recipient_capability: Capability = Capability.EXTERNAL


@dataclass
class PropertyResult:
    """Result of evaluating a single security property."""

    property: SecurityProperty
    satisfied: bool
    reason: str = ""


@dataclass
class FrameworkResult:
    """Aggregate result of all four property checks."""

    results: list[PropertyResult]

    @property
    def all_satisfied(self) -> bool:
        return all(r.satisfied for r in self.results)

    @property
    def failed_properties(self) -> list[SecurityProperty]:
        return [r.property for r in self.results if not r.satisfied]


def evaluate_task_alignment(ctx: SecurityContext) -> PropertyResult:
    """Task Alignment: action must be consistent with stated task."""
    action_lower = ctx.requested_action.lower()
    task_lower = ctx.task_description.lower()

    suspicious = ["delete_all", "exfiltrate", "override", "bypass"]
    for pattern in suspicious:
        if pattern in action_lower and pattern not in task_lower:
            return PropertyResult(
                SecurityProperty.TASK_ALIGNMENT,
                False,
                f"Action '{ctx.requested_action}' not aligned with task",
            )

    if ctx.allowed_actions and ctx.requested_action not in ctx.allowed_actions:
        if not any(a in ctx.requested_action for a in ctx.allowed_actions):
            return PropertyResult(
                SecurityProperty.TASK_ALIGNMENT,
                False,
                f"Action '{ctx.requested_action}' not in allowed actions",
            )

    return PropertyResult(SecurityProperty.TASK_ALIGNMENT, True)


def evaluate_action_alignment(ctx: SecurityContext) -> PropertyResult:
    """Action Alignment: requested action matches intended operation type."""
    external_actions = ("send_email", "post_message", "http_post", "external_send", "transfer")
    action_base = ctx.requested_action.split(":")[0]

    if action_base in external_actions:
        if action_base == "transfer" and "transfer" in ctx.task_description.lower():
            return PropertyResult(SecurityProperty.ACTION_ALIGNMENT, True)
        if "send" not in ctx.task_description.lower() and "email" not in ctx.task_description.lower():
            if "external" not in ctx.task_description.lower():
                return PropertyResult(
                    SecurityProperty.ACTION_ALIGNMENT,
                    False,
                    f"External action '{action_base}' not aligned with task intent",
                )

    return PropertyResult(SecurityProperty.ACTION_ALIGNMENT, True)


def evaluate_source_authorization(ctx: SecurityContext) -> PropertyResult:
    """Source Authorization: data source must be authorized for the action."""
    if ctx.data_provenance.source not in ctx.authorized_sources:
        if ctx.data_provenance.source == Source.TOOL:
            external_actions = ("send_email", "post_message", "http_post", "external_send", "transfer")
            action_base = ctx.requested_action.split(":")[0]
            if action_base in external_actions:
                return PropertyResult(
                    SecurityProperty.SOURCE_AUTHORIZATION,
                    False,
                    "Tool-sourced data not authorized for external action",
                )
    return PropertyResult(SecurityProperty.SOURCE_AUTHORIZATION, True)


def evaluate_data_isolation(ctx: SecurityContext) -> PropertyResult:
    """Data Isolation: untrusted data cannot reach unauthorized recipients."""
    if ctx.data_provenance.is_untrusted():
        if not ctx.data_provenance.can_send_to(ctx.recipient_capability):
            return PropertyResult(
                SecurityProperty.DATA_ISOLATION,
                False,
                f"Untrusted data cannot be sent to {ctx.recipient_capability.value}",
            )
    return PropertyResult(SecurityProperty.DATA_ISOLATION, True)


def evaluate_all_properties(ctx: SecurityContext) -> FrameworkResult:
    """Evaluate all four contextual security properties."""
    results = [
        evaluate_task_alignment(ctx),
        evaluate_action_alignment(ctx),
        evaluate_source_authorization(ctx),
        evaluate_data_isolation(ctx),
    ]
    return FrameworkResult(results=results)
