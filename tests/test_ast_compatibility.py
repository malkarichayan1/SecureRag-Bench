"""Tests for the benign restricted-AST compatibility evaluator."""

from __future__ import annotations

import inspect
from collections import Counter

import pytest

from secure_rag_bench.camel.interpreter import InterpreterError, PolicyHalt
from secure_rag_bench.evaluation import ast_compatibility
from secure_rag_bench.evaluation.ast_compatibility import (
    BenignPlanCase,
    build_benign_plan_catalog,
    evaluate_benign_plan,
    summarize_ast_compatibility,
)
from secure_rag_bench.security.policy import PolicyDecision

EXPECTED_FAMILIES = {"email", "calendar", "repository", "banking", "external_posting"}


def email_case() -> BenignPlanCase:
    return next(case for case in build_benign_plan_catalog() if case.case_id == "email-001")


def _reference_plan(case: BenignPlanCase) -> str:
    """Build the literal plan that satisfies ``case``'s declared expected effect."""
    tool_name = case.expected_effect["tool"]
    match = case.expected_effect["match"]
    signature = inspect.signature(case.tools[tool_name])
    args = ", ".join(repr(match[name]) for name in signature.parameters)
    return f"{tool_name}({args})"


def _broken_transfer_case() -> BenignPlanCase:
    def _always_fails(account: str, amount: float) -> dict[str, str]:
        raise ValueError("simulated ledger outage: account 42 balance -900")

    return BenignPlanCase(
        case_id="banking-broken",
        family="banking",
        user_query="Transfer the deposit to the savings account",
        tools={"transfer": _always_fails},
        expected_effect={"tool": "transfer", "match": {}, "calls": []},
    )


# --- Step 1 acceptance and taxonomy tests (from the plan) -----------------


def test_raw_supported_plan_is_accepted_and_executes_expected_effect() -> None:
    record = evaluate_benign_plan(email_case(), "send_email('team@example.test', 'Status', 'On track')")
    assert record.syntax_accepted
    assert record.execution_succeeded
    assert record.rejection_category is None


@pytest.mark.parametrize(
    ("plan", "category"),
    [
        ("import os", "unsupported_node:Import"),
        ("send_email.upper()", "indirect_call"),
        ("unknown()", "unknown_tool"),
    ],
)
def test_rejections_are_stable(plan: str, category: str) -> None:
    assert evaluate_benign_plan(email_case(), plan).rejection_category == category


# --- Catalog coverage -------------------------------------------------------


def test_catalog_has_at_least_five_cases_per_family_and_unique_ids() -> None:
    catalog = build_benign_plan_catalog()
    case_ids = [case.case_id for case in catalog]
    assert len(case_ids) == len(set(case_ids))

    families = Counter(case.family for case in catalog)
    assert set(families) == EXPECTED_FAMILIES
    assert all(count >= 5 for count in families.values())
    assert all(case.tools for case in catalog)
    assert all(case.user_query.strip() for case in catalog)


@pytest.mark.parametrize("case", build_benign_plan_catalog(), ids=lambda case: case.case_id)
def test_every_catalog_case_accepts_and_executes_its_reference_plan(case: BenignPlanCase) -> None:
    record = evaluate_benign_plan(case, _reference_plan(case))
    assert record.syntax_accepted
    assert record.execution_succeeded
    assert record.rejection_category is None
    assert record.case_id == case.case_id
    assert record.family == case.family


# --- Effect verification beyond "did not crash" -----------------------------


def test_plan_that_runs_without_matching_expected_effect_is_not_a_rejection() -> None:
    record = evaluate_benign_plan(email_case(), "answer = 'noop'")
    assert record.syntax_accepted
    assert not record.execution_succeeded
    assert record.rejection_category is None


def test_repeated_evaluation_of_the_same_case_does_not_leak_prior_tool_calls() -> None:
    """A prior successful call recorded on a case must not leak into a later,
    unrelated evaluation of that same (reused) BenignPlanCase object."""
    case = email_case()

    successful = evaluate_benign_plan(case, "send_email('team@example.test', 'Status', 'On track')")
    assert successful.execution_succeeded

    unrelated = evaluate_benign_plan(case, "answer = 'noop'")
    assert not unrelated.execution_succeeded


# --- Broader rejection taxonomy (beyond the three stable spec examples) ----


def test_tool_runtime_error_is_categorized_as_interpreter_error() -> None:
    case = _broken_transfer_case()
    record = evaluate_benign_plan(case, "transfer('savings-003', 300)")
    assert record.syntax_accepted
    assert not record.execution_succeeded
    assert record.rejection_category == "interpreter_error"
    assert "ledger outage" not in (record.detail or "")


def test_uncaught_runtime_error_is_categorized_as_unexpected_error() -> None:
    # ast.Div is an allowed node, so "1 / 0" passes validate_plan but blows up
    # inside the interpreter's own binop evaluation, uncaught by InterpreterError.
    record = evaluate_benign_plan(email_case(), "1 / 0")
    assert record.syntax_accepted
    assert not record.execution_succeeded
    assert record.rejection_category == "unexpected_error:ZeroDivisionError"


def test_categorize_maps_policy_halt_to_stable_category_with_redacted_detail() -> None:
    # PolicyHalt can't currently be reached through the live check_policy() wiring
    # (requires_user_confirmation is never set True there), so the taxonomy branch
    # is verified directly against the categorization helper.
    decision = PolicyDecision(allowed=False, reason="tool-output-with-secret", requires_user_confirmation=True)
    category, detail = ast_compatibility._categorize(PolicyHalt("halt", decision))
    assert category == "policy_halt"
    assert "tool-output-with-secret" not in detail


def test_categorize_maps_generic_interpreter_error() -> None:
    category, _detail = ast_compatibility._categorize(InterpreterError("Plan produced no result"))
    assert category == "interpreter_error"


def test_categorize_maps_unexpected_exception_types() -> None:
    category, detail = ast_compatibility._categorize(ZeroDivisionError("division by zero"))
    assert category == "unexpected_error:ZeroDivisionError"
    assert "division by zero" not in detail


# --- Grouped summary ---------------------------------------------------------


def test_summarize_ast_compatibility_reports_totals_and_family_breakdown() -> None:
    catalog = build_benign_plan_catalog()
    records = [evaluate_benign_plan(case, _reference_plan(case)) for case in catalog]
    records.append(evaluate_benign_plan(email_case(), "import os"))

    summary = summarize_ast_compatibility(records)

    assert summary["case_count"] == len(records)
    assert summary["accepted"] == len(catalog)
    assert summary["execution_successes"] == len(catalog)
    wilson = summary["acceptance_wilson_95"]
    assert 0.0 <= wilson["lower"] <= wilson["upper"] <= 1.0
    assert set(summary["by_family"]) == EXPECTED_FAMILIES
    for family, family_summary in summary["by_family"].items():
        assert family_summary["count"] >= 5
        if family == "email":
            # The appended "import os" rejection also belongs to the email family.
            assert family_summary["accepted"] == family_summary["count"] - 1
        else:
            assert family_summary["accepted"] == family_summary["count"]
    assert summary["rejection_categories"]["unsupported_node:Import"] == 1


def test_summarize_ast_compatibility_rejects_empty_records() -> None:
    with pytest.raises(ValueError):
        summarize_ast_compatibility([])
