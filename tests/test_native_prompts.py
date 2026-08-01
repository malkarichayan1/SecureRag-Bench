from secure_rag_bench.evaluation.native_prompts import (
    PromptCondition,
    apply_prompt_contract,
)


def test_strict_react_contract_requires_one_available_action() -> None:
    rendered = apply_prompt_contract(
        "BASE", PromptCondition.STRICT_REACT, ["GmailReadEmail", "GmailSendEmail"]
    )

    assert "exactly one Action:" in rendered
    assert "Do not invent an Observation" in rendered
    assert "eval" not in rendered and "ASR" not in rendered


def test_original_contract_is_byte_preserving() -> None:
    assert apply_prompt_contract("BASE", PromptCondition.ORIGINAL, ["ToolA"]) == "BASE"


def test_structured_contract_allows_a_final_answer_without_an_action() -> None:
    rendered = apply_prompt_contract(
        "BASE", PromptCondition.STRUCTURED_SINGLE_ACTION, ["GmailReadEmail"]
    )

    assert "Final Answer: <plain answer to the user>" in rendered
    assert "If you use a tool, take exactly one Action:" in rendered
    assert "Use exactly one Action:" not in rendered
