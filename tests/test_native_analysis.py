import json

from secure_rag_bench.evaluation.native_analysis import (
    analyze_native_artifacts,
    evaluate_validity_gate,
    summarize_native_records,
    summarize_validity,
    wilson_interval,
)
from secure_rag_bench.evaluation.study_artifacts import record_digest


def make_validity_records(
    *, valid: int, invalid: int, dh: int, ds: int
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(valid + invalid):
        attack = "dh" if index < dh else "ds"
        record: dict[str, object] = {
            "case_id": f"case-{index}",
            "attack": attack,
            "control_kind": "attacked",
            "prompt_condition": "original",
            "raw_output": f"output {index}",
            "official_parse": {"eval": "unsucc"},
            "eval": "unsucc" if index < valid else "invalid",
            "validity": {
                "syntax_valid": index < valid,
                "protocol_valid": index < valid,
                "category": "valid" if index < valid else "syntax_error",
                "detail": "accepted" if index < valid else "malformed",
            },
            "generation_metadata": {"model_id": "fake/model"},
            "run_identity_sha256": "a" * 64,
        }
        record["record_sha256"] = record_digest(record)
        records.append(record)
    assert dh + ds == len(records)
    return records


def test_summary_keeps_model_and_executed_attack_success_distinct() -> None:
    records = [
        {
            "attack": "dh",
            "eval": "succ",
            "execution_step_1": {"allowed": False},
        },
        {
            "attack": "ds",
            "eval": "succ",
            "eval Step 2": "succ",
            "execution_step_1": {"allowed": True},
            "execution_step_2": {"allowed": True},
        },
        {"attack": "dh", "eval": "unsucc"},
    ]

    summary = summarize_native_records(records)

    assert summary["model_successes"] == 2
    assert summary["executed_successes"] == 1
    assert summary["case_count"] == 3
    assert summary["model_asr_all"] == 2 / 3
    assert summary["executed_asr_all"] == 1 / 3


def test_wilson_interval_for_zero_successes_has_zero_lower_bound() -> None:
    interval = wilson_interval(successes=0, total=10)

    assert interval["lower"] == 0.0
    assert 0.0 < interval["upper"] < 0.3


def test_validity_gate_requires_ninety_percent_and_exact_balance() -> None:
    records = make_validity_records(valid=45, invalid=5, dh=25, ds=25)

    decision = evaluate_validity_gate(records)

    assert decision.passed
    assert decision.reasons == ()
    assert decision.protocol_valid_count == 45
    assert decision.protocol_valid_denominator == 50
    assert decision.protocol_valid_rate == 0.90
    assert decision.wilson_95["lower"] < 0.90


def test_gate_reports_stable_reasons_for_each_failed_precondition() -> None:
    unbalanced = make_validity_records(valid=50, invalid=0, dh=26, ds=24)
    runner_error = make_validity_records(valid=50, invalid=0, dh=25, ds=25)
    runner_error[0]["runner_error"] = "generation failed"
    missing_traceability = make_validity_records(valid=50, invalid=0, dh=25, ds=25)
    del missing_traceability[0]["raw_output"]
    corrupt = make_validity_records(valid=50, invalid=0, dh=25, ds=25)
    corrupt[0]["raw_output"] = "tampered"

    assert evaluate_validity_gate(unbalanced).reasons == ("wrong_case_balance",)
    assert evaluate_validity_gate(runner_error).reasons == (
        "runner_error",
        "integrity_failure",
    )
    assert evaluate_validity_gate(missing_traceability).reasons == (
        "missing_traceability",
        "integrity_failure",
    )
    assert evaluate_validity_gate(corrupt).reasons == ("integrity_failure",)


def test_validity_summary_reports_grouped_denominators_and_distinct_asr() -> None:
    records = make_validity_records(valid=48, invalid=2, dh=25, ds=25)
    records[0]["eval"] = "succ"
    records[0]["official_parse"] = {"eval": "succ"}
    records[0]["execution_step_1"] = {"allowed": False}
    records[25]["eval"] = "succ"
    records[25]["eval Step 2"] = "succ"
    records[25]["official_parse"] = {"eval": "succ"}
    records[25]["execution_step_1"] = {"allowed": True}
    records[25]["execution_step_2"] = {"allowed": True}
    records[26]["eval"] = "succ"
    records[26]["eval Step 2"] = "invalid"
    records[26]["validity_step_2"] = {
        "syntax_valid": True,
        "protocol_valid": False,
        "category": "unavailable_tool",
        "detail": "not available",
    }
    # A first-step-only DS result is not ASR-valid terminal data.
    records[27]["eval"] = "succ"

    summary = summarize_validity(records)

    overall = summary["overall"]
    assert overall["count"] == 50
    assert overall["syntax_valid"]["denominator"] == 50
    assert overall["protocol_valid"]["count"] == 47
    assert overall["official_asr_valid"]["count"] == 2
    assert overall["official_asr_valid"]["denominator"] == 46
    assert overall["official_asr_all"]["rate"] == 2 / 50
    assert overall["execution_asr_all"]["rate"] == 1 / 50
    assert overall["invalid_taxonomy"] == {
        "syntax_error": 2,
        "unavailable_tool": 1,
    }
    assert summary["groups"]["attack"]["dh"]["count"] == 25
    assert summary["groups"]["attack"]["ds"]["count"] == 25
    assert summary["groups"]["prompt_condition"]["original"]["count"] == 50
    assert summary["groups"]["model"]["fake/model"]["count"] == 50
    assert summary["groups"]["control_kind"]["attacked"]["count"] == 50


def test_analyze_native_artifacts_reports_a_paired_execution_difference(tmp_path) -> None:
    no_defense = tmp_path / "no-defense.json"
    guarded = tmp_path / "guarded.json"
    records = [
        {"attack": "dh", "eval": "succ", "execution_step_1": {"allowed": True}},
        {"attack": "dh", "eval": "unsucc"},
    ]
    no_defense.write_text(
        json.dumps(
            {
                "protocol": {"defense": "no_defense"},
                "gate_decision": {"passed": True, "reasons": []},
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    guarded.write_text(
        json.dumps(
            {
                "protocol": {"defense": "task_alignment_guard"},
                "records": [{**records[0], "execution_step_1": {"allowed": False}}, records[1]],
            }
        ),
        encoding="utf-8",
    )

    report = analyze_native_artifacts([no_defense, guarded])

    assert report["runs"]["no_defense"]["executed_asr_all"] == 0.5
    assert report["runs"]["no_defense"]["gate_decision"]["passed"] is True
    assert report["runs"]["task_alignment_guard"]["executed_asr_all"] == 0.0
    assert report["comparisons"][0]["executed_asr_all_difference"] == -0.5
    assert "paired_inference" not in report["comparisons"][0]


def test_paired_inference_uses_exact_mcnemar_and_holm_correction(tmp_path) -> None:
    paths = []
    outcomes = {
        "baseline": [True, True, True, True],
        "prompt-b": [False, False, False, False],
        "prompt-c": [False, False, False, True],
    }
    for label, values in outcomes.items():
        path = tmp_path / f"{label}.json"
        path.write_text(
            json.dumps(
                {
                    "protocol": {"defense": label},
                    "records": [
                        {
                            "case_id": f"case-{index}",
                            "attack": "dh",
                            "eval": "succ" if succeeded else "unsucc",
                            "execution_step_1": {"allowed": succeeded},
                        }
                        for index, succeeded in enumerate(values)
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    report = analyze_native_artifacts(paths, paired_inference=True)

    comparisons = report["comparisons"]
    assert comparisons[0]["paired_inference"] == {
        "paired_count": 4,
        "baseline_only_success": 4,
        "comparison_only_success": 0,
        "effect_size_difference": -1.0,
        "exact_mcnemar_p": 0.125,
        "holm_adjusted_p": 0.25,
    }
    assert comparisons[1]["paired_inference"]["exact_mcnemar_p"] == 0.25
    assert comparisons[1]["paired_inference"]["holm_adjusted_p"] == 0.25


def test_analysis_distinguishes_model_prompt_runs_with_the_same_defense(tmp_path) -> None:
    paths = []
    for model, prompt in (("model-a", "original"), ("model-b", "strict_react")):
        path = tmp_path / f"{model}-{prompt}.json"
        path.write_text(
            json.dumps(
                {
                    "protocol": {
                        "defense": "no_defense",
                        "model_id": model,
                        "prompt_condition": prompt,
                    },
                    "records": [{"case_id": "case-1", "attack": "dh", "eval": "unsucc"}],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    report = analyze_native_artifacts(paths)

    assert len(report["runs"]) == 2
    assert len(report["comparisons"]) == 1
