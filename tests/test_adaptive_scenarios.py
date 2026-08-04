"""Tests for the typed adaptive attack scenario catalog."""

from __future__ import annotations

import codecs

import pytest

from secure_rag_bench.camel.interpreter import CaMeLInterpreter, PolicyHalt, SecurityViolation
from secure_rag_bench.camel.quarantined_llm import MockQuarantinedLLM
from secure_rag_bench.evaluation.adaptive_scenarios import AdaptiveScenario, build_adaptive_scenarios
from secure_rag_bench.security.policy import EXTERNAL_ACTIONS

EXPECTED_ATTACK_FAMILIES = {"cem_integrated", "multi_step", "split_obfuscated", "policy_aware"}
CEM_PREFIXES = ["optimized prefix"]


# --- Step 1: required catalog coverage test (verbatim from the plan) -------


def test_catalog_has_all_families_configurations_and_pairs() -> None:
    scenarios = build_adaptive_scenarios(["optimized prefix"])
    assert {s.family for s in scenarios if s.is_attack} == {
        "cem_integrated", "multi_step", "split_obfuscated", "policy_aware"
    }
    attacks = {s.pair_id for s in scenarios if s.is_attack}
    controls = {s.pair_id for s in scenarios if not s.is_attack}
    assert attacks == controls


# --- Catalog shape and coverage ---------------------------------------------


def test_catalog_covers_all_eight_required_concepts_with_min_two_policy_pairs() -> None:
    scenarios = build_adaptive_scenarios(CEM_PREFIXES)
    attacks = [s for s in scenarios if s.is_attack]

    by_family: dict[str, list[AdaptiveScenario]] = {}
    for scenario in attacks:
        by_family.setdefault(scenario.family, []).append(scenario)

    assert len(by_family["cem_integrated"]) == 1
    assert len(by_family["multi_step"]) == 2
    assert len(by_family["split_obfuscated"]) == 4
    assert len(by_family["policy_aware"]) >= 2


def test_scenario_ids_are_globally_unique() -> None:
    scenarios = build_adaptive_scenarios(CEM_PREFIXES)
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))


def test_every_scenario_has_a_non_empty_user_query() -> None:
    scenarios = build_adaptive_scenarios(CEM_PREFIXES)
    assert all(s.user_query.strip() for s in scenarios)


def test_cem_prefix_is_injected_into_the_cem_integrated_document() -> None:
    scenarios = build_adaptive_scenarios(CEM_PREFIXES)
    cem_attack = next(s for s in scenarios if s.family == "cem_integrated" and s.is_attack)
    assert any(CEM_PREFIXES[0] in doc.text for doc in cem_attack.documents)


def test_build_adaptive_scenarios_rejects_empty_cem_prefixes() -> None:
    with pytest.raises(ValueError):
        build_adaptive_scenarios([])


def test_multi_document_split_scenarios_have_exactly_two_documents() -> None:
    scenarios = build_adaptive_scenarios(CEM_PREFIXES)
    split_pairs = {"split_instruction_reconstructed", "split_value_transfer"}
    for scenario in scenarios:
        if scenario.pair_id in split_pairs and scenario.is_attack:
            assert len(scenario.documents) == 2


# --- target_effect oracle shape ---------------------------------------------


def test_every_target_effect_names_a_valid_external_action_tool() -> None:
    scenarios = build_adaptive_scenarios(CEM_PREFIXES)
    for scenario in scenarios:
        assert scenario.target_effect["tool"] in EXTERNAL_ACTIONS


def test_every_target_effect_sets_exactly_one_check_kind() -> None:
    scenarios = build_adaptive_scenarios(CEM_PREFIXES)
    for scenario in scenarios:
        has_contains = "argument_contains" in scenario.target_effect
        has_equals = "argument_equals" in scenario.target_effect
        assert has_contains != has_equals, scenario.scenario_id


def test_target_effect_tool_name_appears_as_a_call_in_the_plan() -> None:
    scenarios = build_adaptive_scenarios(CEM_PREFIXES)
    for scenario in scenarios:
        tool = scenario.target_effect["tool"]
        assert f"{tool}(" in scenario.plan, scenario.scenario_id


# --- Step 4: plan language validity, unique IDs, paired tool parity --------


def _all_known_tools() -> dict[str, object]:
    """A superset tools dict covering every tool name any catalog plan calls.

    validate_plan only checks that a Call node's function name is a key in
    this dict (or the special-cased quarantine_parse); it does not care
    whether extra, unused keys are present, so one shared superset dict is
    sufficient to validate every scenario's plan.
    """

    class _Toolbox:
        def send_email(self, recipient: str, subject: str, body: object) -> dict:
            return {"recipient": recipient, "subject": subject, "body": body}

        def transfer(self, account: object, amount: float) -> dict:
            return {"account": account, "amount": amount}

        def post_message(self, channel: str, text: object) -> dict:
            return {"channel": channel, "text": text}

        def retrieve(self, query: str) -> str:
            return f"retrieved:{query}"

        def retrieve_doc_a(self, query: str) -> str:
            return f"retrieved-a:{query}"

        def retrieve_doc_b(self, query: str) -> str:
            return f"retrieved-b:{query}"

        def decode(self, text: str) -> str:
            return codecs.encode(str(text), "rot_13")

    toolbox = _Toolbox()
    return {
        name: getattr(toolbox, name)
        for name in (
            "send_email",
            "transfer",
            "post_message",
            "retrieve",
            "retrieve_doc_a",
            "retrieve_doc_b",
            "decode",
        )
    }


@pytest.mark.parametrize(
    "scenario", build_adaptive_scenarios(CEM_PREFIXES), ids=lambda s: s.scenario_id
)
def test_every_plan_passes_validate_plan_with_its_declared_tools(scenario: AdaptiveScenario) -> None:
    interpreter = CaMeLInterpreter(tools=_all_known_tools())
    interpreter.validate_plan(scenario.plan)


def test_attack_control_pairs_target_the_same_external_tool() -> None:
    scenarios = build_adaptive_scenarios(CEM_PREFIXES)
    by_pair: dict[str, list[AdaptiveScenario]] = {}
    for scenario in scenarios:
        by_pair.setdefault(scenario.pair_id, []).append(scenario)

    for pair_id, members in by_pair.items():
        assert len(members) == 2, pair_id
        attack = next(s for s in members if s.is_attack)
        control = next(s for s in members if not s.is_attack)
        assert attack.target_effect["tool"] == control.target_effect["tool"], pair_id


def test_attack_control_pairs_have_comparable_plan_complexity() -> None:
    """Controls should not be trivial one-liners next to a multi-statement attack."""
    scenarios = build_adaptive_scenarios(CEM_PREFIXES)
    by_pair: dict[str, list[AdaptiveScenario]] = {}
    for scenario in scenarios:
        by_pair.setdefault(scenario.pair_id, []).append(scenario)

    for pair_id, members in by_pair.items():
        attack = next(s for s in members if s.is_attack)
        control = next(s for s in members if not s.is_attack)
        attack_statements = attack.plan.count("\n") + 1
        control_statements = control.plan.count("\n") + 1
        assert abs(attack_statements - control_statements) <= 1, pair_id


# --- Optional smoke test: representative execution against stub tools ------
#
# Not required by the plan's Step 4 (which only asks for validate_plan), but
# executes every scenario end-to-end with stub tools to raise confidence that
# Task 3's runner will find these plans genuinely executable. Controls MUST
# execute successfully and achieve their declared effect (they are meant to
# be legitimate, authorized actions). Attacks may be blocked by policy
# (PolicyHalt / SecurityViolation) -- that is a defense holding, not a test
# failure -- but must not raise any other, unexpected error.


def _toolbox_and_tools(scenario: AdaptiveScenario):
    calls: list[tuple[str, dict]] = []

    class _Toolbox:
        def send_email(self, recipient: str, subject: str, body: object) -> dict:
            calls.append(("send_email", {"recipient": recipient, "subject": subject, "body": body}))
            return {"status": "ok"}

        def transfer(self, account: object, amount: float) -> dict:
            calls.append(("transfer", {"account": account, "amount": amount}))
            return {"status": "ok"}

        def post_message(self, channel: str, text: object) -> dict:
            calls.append(("post_message", {"channel": channel, "text": text}))
            return {"status": "ok"}

    docs = scenario.documents

    def retrieve(query: str) -> str:
        return docs[0].text if docs else ""

    def retrieve_doc_a(query: str) -> str:
        return docs[0].text

    def retrieve_doc_b(query: str) -> str:
        return docs[1].text

    def decode(text: str) -> str:
        return codecs.encode(str(text), "rot_13")

    toolbox = _Toolbox()
    tools = {
        "send_email": toolbox.send_email,
        "transfer": toolbox.transfer,
        "post_message": toolbox.post_message,
        "retrieve": retrieve,
        "retrieve_doc_a": retrieve_doc_a,
        "retrieve_doc_b": retrieve_doc_b,
        "decode": decode,
    }
    return calls, tools


def _effect_matches(target_effect: dict, calls: list[tuple[str, dict]]) -> bool:
    tool = target_effect["tool"]
    for call_tool, arguments in calls:
        if call_tool != tool:
            continue
        if "argument_contains" in target_effect:
            needle = target_effect["argument_contains"]
            if any(needle in str(value) for value in arguments.values()):
                return True
        elif "argument_equals" in target_effect:
            expected = target_effect["argument_equals"]
            if all(arguments.get(key) == value for key, value in expected.items()):
                return True
    return False


@pytest.mark.parametrize(
    "scenario", build_adaptive_scenarios(CEM_PREFIXES), ids=lambda s: s.scenario_id
)
def test_smoke_execute_every_scenario_plan(scenario: AdaptiveScenario) -> None:
    calls, tools = _toolbox_and_tools(scenario)
    interpreter = CaMeLInterpreter(tools=tools, quarantined_llm=MockQuarantinedLLM())

    try:
        interpreter.execute(scenario.plan, user_query=scenario.user_query)
    except (PolicyHalt, SecurityViolation):
        if not scenario.is_attack:
            pytest.fail(f"benign control {scenario.scenario_id} was blocked by policy")
        return

    if not scenario.is_attack:
        assert _effect_matches(scenario.target_effect, calls), scenario.scenario_id
