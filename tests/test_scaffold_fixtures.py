from secure_rag_bench.rag.hybrid_retrieval import Document


def test_deterministic_corpus_contains_benign_and_injection_documents(deterministic_corpus) -> None:
    assert [document.doc_id for document in deterministic_corpus] == [
        "status-001",
        "status-002",
        "injection-001",
    ]
    assert all(isinstance(document, Document) for document in deterministic_corpus)
    assert "ignore previous instructions" in deterministic_corpus[-1].text.lower()


def test_simulated_tools_record_drafts_and_sends(simulated_tools) -> None:
    draft = simulated_tools.create_email_draft("alice@example.test", "Status", "On track")
    sent = simulated_tools.send_email("alice@example.test", "Status", "On track")

    assert draft["status"] == "draft"
    assert sent["status"] == "sent"
    assert [call.name for call in simulated_tools.calls] == [
        "create_email_draft",
        "send_email",
    ]
