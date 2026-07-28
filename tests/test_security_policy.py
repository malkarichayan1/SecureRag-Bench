from secure_rag_bench.camel.provenance import TrackedValue
from secure_rag_bench.security.policy import check_policy


def test_external_send_of_tool_data_is_denied_even_after_confirmation() -> None:
    decision = check_policy(
        task_description="Send Alice a project update",
        action="send_email:alice@example.test",
        data_provenance=TrackedValue.from_tool("ignore safety").provenance,
        user_confirmed=True,
    )

    assert not decision.allowed
    assert decision.failed_property == "data_isolation"
    assert not decision.requires_user_confirmation


def test_user_derived_external_send_aligned_with_task_is_allowed() -> None:
    decision = check_policy(
        task_description="Send Alice a project update",
        action="send_email:alice@example.test",
        data_provenance=TrackedValue.from_user("Update").provenance,
    )

    assert decision.allowed


def test_provenance_ablation_allows_tainted_data_when_action_is_authorized() -> None:
    decision = check_policy(
        task_description="Send Alice a project update",
        action="send_email:alice@example.test",
        data_provenance=TrackedValue.from_tool("attacker content").provenance,
        enforce_provenance=False,
    )

    assert decision.allowed
