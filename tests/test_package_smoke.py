from pathlib import Path

from secure_rag_bench import CaMeLInterpreter, MockPrivilegedLLM, MockQuarantinedLLM
from secure_rag_bench.camel.quarantined_llm import ParsedRetrieval, wrap_untrusted


def test_public_package_exports_offline_components() -> None:
    assert CaMeLInterpreter is not None
    assert MockPrivilegedLLM is not None
    assert MockQuarantinedLLM is not None


def test_mock_components_work_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    plan = MockPrivilegedLLM().generate_plan("find project status")
    parsed = MockQuarantinedLLM().parse(
        wrap_untrusted("Status: on track"),
        "ParsedRetrieval",
    )

    assert "retrieve" in plan
    assert isinstance(parsed, ParsedRetrieval)


def test_readme_documents_free_baselines_as_non_official() -> None:
    readme = " ".join(Path("README.md").read_text(encoding="utf-8").lower().split())

    assert "injecagent-baselines" in readme
    assert "deterministic surrogates" in readme
    assert "not an official injecagent score" in readme


def test_readme_documents_kaggle_gate_resume_and_bundle_import() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for phrase in ("90% validity gate", "resume", "export_study_bundle.py", "import_study_bundle.py"):
        assert phrase in readme
