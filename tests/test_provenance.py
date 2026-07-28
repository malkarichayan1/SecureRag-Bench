from secure_rag_bench.camel.provenance import Capability, Source, TrackedValue


def test_merging_user_and_tool_values_keeps_both_sources_and_removes_external_reader() -> None:
    merged = TrackedValue.from_user("Alice").merge_provenance(
        TrackedValue.from_tool("attacker text")
    )

    assert merged.sources == frozenset({Source.USER, Source.TOOL})
    assert merged.allowed_readers == frozenset({Capability.INTERNAL})
    assert merged.is_tainted()
    assert not merged.is_externally_readable()


def test_tool_value_is_never_externally_readable_by_default() -> None:
    assert not TrackedValue.from_tool("retrieved").provenance.is_externally_readable()
