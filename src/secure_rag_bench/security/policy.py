"""Policy enforcement before tool execution."""

from __future__ import annotations

from dataclasses import dataclass

from secure_rag_bench.camel.provenance import Capability, Provenance, Source
from secure_rag_bench.security.framework import SecurityContext, SecurityProperty, evaluate_all_properties


@dataclass
class PolicyDecision:
    """Result of a policy check."""

    allowed: bool
    reason: str = ""
    requires_user_confirmation: bool = False
    failed_property: str | None = None


EXTERNAL_ACTIONS = frozenset({"send_email", "post_message", "http_post", "external_send", "transfer"})


def check_policy(
    *,
    task_description: str,
    action: str,
    data_provenance: Provenance,
    user_confirmed: bool = False,
    allowed_actions: frozenset[str] | None = None,
    enforce_provenance: bool = True,
) -> PolicyDecision:
    """
    Check policy before every tool execution.

    Tainted data is never eligible for an external capability. Confirmation may
    approve a future untainted action, but cannot declassify retrieved content.
    """
    action_base = action.split(":")[0]
    is_external = action_base in EXTERNAL_ACTIONS
    recipient_capability = Capability.EXTERNAL if is_external else Capability.INTERNAL

    ctx = SecurityContext(
        task_description=task_description,
        requested_action=action,
        data_provenance=data_provenance,
        authorized_sources=frozenset({Source.USER, Source.TOOL}),
        allowed_actions=allowed_actions,
        recipient_capability=recipient_capability,
    )

    framework_result = evaluate_all_properties(ctx)

    failed_results = [r for r in framework_result.results if not r.satisfied]
    if not enforce_provenance:
        failed_results = [
            result
            for result in failed_results
            if result.property not in {
                SecurityProperty.SOURCE_AUTHORIZATION,
                SecurityProperty.DATA_ISOLATION,
            }
        ]

    if failed_results:
        failed = failed_results[0]

        if failed.property.value == "data_isolation" and is_external:
            return PolicyDecision(
                allowed=False,
                reason="Untrusted data cannot be sent to external recipient.",
                failed_property=failed.property.value,
            )

        if failed.property.value in ("task_alignment", "action_alignment", "source_authorization"):
            return PolicyDecision(
                allowed=False,
                reason=failed.reason,
                failed_property=failed.property.value,
            )

        return PolicyDecision(
            allowed=False,
            reason=failed.reason,
            failed_property=failed.property.value,
        )

    return PolicyDecision(allowed=True, reason="Policy check passed")
