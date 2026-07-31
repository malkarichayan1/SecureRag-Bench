import json

from secure_rag_bench.evaluation.native_analysis import (
    analyze_native_artifacts,
    summarize_native_records,
    wilson_interval,
)


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


def test_analyze_native_artifacts_reports_a_paired_execution_difference(tmp_path) -> None:
    no_defense = tmp_path / "no-defense.json"
    guarded = tmp_path / "guarded.json"
    records = [
        {"attack": "dh", "eval": "succ", "execution_step_1": {"allowed": True}},
        {"attack": "dh", "eval": "unsucc"},
    ]
    no_defense.write_text(
        json.dumps({"protocol": {"defense": "no_defense"}, "records": records}),
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
    assert report["runs"]["task_alignment_guard"]["executed_asr_all"] == 0.0
    assert report["comparisons"][0]["executed_asr_all_difference"] == -0.5
