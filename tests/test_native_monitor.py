import pytest


def test_task_alignment_gate_blocks_an_attacker_tool() -> None:
    from secure_rag_bench.evaluation.native_monitor import gate_native_action

    decision = gate_native_action(
        proposed_tool="GmailSendEmail",
        user_tool="GmailReadEmail",
        defense="task_alignment_guard",
    )

    assert not decision.allowed
    assert decision.failed_property == "task_alignment"


def test_task_alignment_gate_allows_the_requested_user_tool() -> None:
    from secure_rag_bench.evaluation.native_monitor import gate_native_action

    decision = gate_native_action(
        proposed_tool="GmailReadEmail",
        user_tool="GmailReadEmail",
        defense="task_alignment_guard",
    )

    assert decision.allowed


def test_no_defense_allows_an_attacker_tool() -> None:
    from secure_rag_bench.evaluation.native_monitor import gate_native_action

    decision = gate_native_action(
        proposed_tool="GmailSendEmail",
        user_tool="GmailReadEmail",
        defense="no_defense",
    )

    assert decision.allowed


def test_unknown_native_defense_is_rejected() -> None:
    from secure_rag_bench.evaluation.native_monitor import gate_native_action

    with pytest.raises(ValueError, match="unsupported native defense"):
        gate_native_action(
            proposed_tool="GmailSendEmail",
            user_tool="GmailReadEmail",
            defense="full_monitor",
        )
