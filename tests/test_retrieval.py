import pytest

from secure_rag_bench.rag.hybrid_retrieval import HybridRetriever
from secure_rag_bench.rag.pipeline import SecureRAGPipeline


def test_hybrid_retrieval_returns_requested_number_of_descending_results(deterministic_corpus) -> None:
    results = HybridRetriever(deterministic_corpus).retrieve("project status", top_k=2)

    assert len(results) == 2
    assert results[0].combined_score >= results[1].combined_score


def test_hybrid_retriever_rejects_invalid_weight_totals(deterministic_corpus) -> None:
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(deterministic_corpus, semantic_weight=0.8, bm25_weight=0.8)


def test_pipeline_sends_only_xml_delimited_content_to_quarantined_parser(deterministic_corpus) -> None:
    result = SecureRAGPipeline(deterministic_corpus).run("project status")

    assert result.untrusted_bundle.startswith("<untrusted_content>")
    assert "<document" in result.untrusted_bundle
