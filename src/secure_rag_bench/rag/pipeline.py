"""Full secure RAG pipeline with retrieval barrier and XML delimiters."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from secure_rag_bench.camel.quarantined_llm import QuarantinedLLM, MockQuarantinedLLM, wrap_untrusted
from secure_rag_bench.rag.hybrid_retrieval import Document, HybridRetriever, MockEmbeddingModel
from secure_rag_bench.rag.reranker import NeuralReranker, RerankedResult


@dataclass
class PipelineConfig:
    """Configuration for the secure RAG pipeline."""

    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    parse_schema: str = "ParsedRetrieval"


@dataclass
class PipelineResult:
    """Result of a secure RAG pipeline run."""

    query: str
    retrieved_count: int
    reranked: list[RerankedResult] = field(default_factory=list)
    untrusted_bundle: str = ""
    parsed: BaseModel | None = None


class SecureRAGPipeline:
    """
    Multi-stage secure RAG pipeline:
    1. Hybrid retrieval (semantic + BM25) → top-20
    2. Neural reranking (cross-encoder) → top-5
    3. Wrap in XML delimiters as untrusted
    4. Parse via Quarantined LLM (no tool access)
    """

    def __init__(
        self,
        documents: list[Document],
        quarantined_llm: QuarantinedLLM | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.retriever = HybridRetriever(documents, embedding_model=MockEmbeddingModel())
        self.reranker = NeuralReranker(output_top_k=self.config.rerank_top_k)
        self.quarantined_llm = quarantined_llm or MockQuarantinedLLM()

    def run(self, query: str) -> PipelineResult:
        """Execute the full secure RAG pipeline."""
        candidates = self.retriever.retrieve(query, top_k=self.config.retrieval_top_k)
        reranked = self.reranker.rerank(query, candidates)
        untrusted_bundle = self._build_untrusted_bundle(reranked)
        parsed = self.quarantined_llm.parse(untrusted_bundle, self.config.parse_schema)

        return PipelineResult(
            query=query,
            retrieved_count=len(candidates),
            reranked=reranked,
            untrusted_bundle=untrusted_bundle,
            parsed=parsed,
        )

    def _build_untrusted_bundle(self, reranked: list[RerankedResult]) -> str:
        """Combine reranked documents into XML-delimited untrusted bundle."""
        parts = []
        for i, result in enumerate(reranked):
            doc = result.retrieval_result.document
            parts.append(
                f"<document id=\"{doc.doc_id}\" rank=\"{i + 1}\">\n"
                f"<title>{doc.metadata.get('title', doc.doc_id)}</title>\n"
                f"<content>{doc.text}</content>\n"
                f"</document>"
            )
        combined = "\n\n".join(parts)
        return wrap_untrusted(combined)

    def retrieve_only(self, query: str) -> list[RerankedResult]:
        """Run retrieval + reranking without quarantined parsing."""
        candidates = self.retriever.retrieve(query, top_k=self.config.retrieval_top_k)
        return self.reranker.rerank(query, candidates)
