from __future__ import annotations

import hashlib
import json

import pytest

from secure_rag_bench.evaluation.native_cases import (
    NativeCase,
    build_validity_split,
    load_native_cases,
)


def make_cases(*, dh: int, ds: int, tool_families: int) -> list[NativeCase]:
    cases: list[NativeCase] = []
    for attack, count in (("dh", dh), ("ds", ds)):
        for index in range(count):
            item = {
                "User Tool": f"Toolkit{index % tool_families}Tool",
                "User Instruction": f"{attack} instruction {index}",
                "Tool Response": f"{attack} response {index}",
            }
            canonical = json.dumps(
                {
                    "attack": attack,
                    "User Instruction": item["User Instruction"],
                    "Tool Response": item["Tool Response"],
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            cases.append(
                NativeCase(
                    attack=attack,
                    item=item,
                    case_id=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                )
            )
    return cases


def test_validity_split_is_disjoint_balanced_and_deterministic() -> None:
    cases = make_cases(dh=80, ds=80, tool_families=5)
    calibration_ids = {case.case_id for case in cases[:50]}

    first = build_validity_split(cases, calibration_ids, seed=20260801)
    second = build_validity_split(cases, calibration_ids, seed=20260801)

    assert [c.case_id for c in first["held_out"]] == [c.case_id for c in second["held_out"]]
    assert len(first["held_out"]) == 50
    assert sum(c.attack == "dh" for c in first["held_out"]) == 25
    assert not calibration_ids & {c.case_id for c in first["held_out"]}


def test_load_native_cases_uses_canonical_sha256_ids(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    item = {
        "User Tool": "MailRead",
        "User Instruction": "Read my mail",
        "Tool Response": "A response",
    }
    for attack in ("dh", "ds"):
        (data / f"test_cases_{attack}_base.json").write_text(
            json.dumps([item]), encoding="utf-8"
        )

    loaded = load_native_cases(tmp_path, "base")

    expected = hashlib.sha256(
        b'{"Tool Response":"A response","User Instruction":"Read my mail","attack":"dh"}'
    ).hexdigest()
    assert loaded[0].case_id == expected
    assert [case.attack for case in loaded] == ["dh", "ds"]


def test_validity_split_rejects_insufficient_eligible_cases() -> None:
    cases = make_cases(dh=24, ds=25, tool_families=2)

    with pytest.raises(ValueError, match="25 eligible cases"):
        build_validity_split(cases, set())
