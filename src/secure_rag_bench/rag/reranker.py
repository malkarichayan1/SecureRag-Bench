"""Neural reranker: cross-encoder top-20 → top-5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from secure_rag_bench.rag.hybrid_retrieval import RetrievalResult


@runtime_checkable
class CrossEncoderModel(Protocol):
    """Protocol for cross-encoder reranking models."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score query-document pairs."""
        ...


class MockCrossEncoder:
    """Deterministic mock cross-encoder for tests."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_tokens = set(query.lower().split())
        scores = []
        for doc in documents:
            doc_tokens = set(doc.lower().split())
            overlap = len(query_tokens & doc_tokens)
            length_penalty = 1.0 / (1.0 + len(doc) / 1000.0)
            scores.append(overlap * 0.5 + length_penalty * 0.3 + 0.2)
        return scores


class SentenceTransformerCrossEncoder:
    """Real cross-encoder using sentence-transformers (optional)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


@dataclass
class RerankedResult:
    """A reranked retrieval result."""

    retrieval_result: RetrievalResult
    rerank_score: float


class NeuralReranker:
    """Cross-encoder reranking: takes top-20, returns top-5."""

    def __init__(
        self,
        cross_encoder: CrossEncoderModel | None = None,
        input_top_k: int = 20,
        output_top_k: int = 5,
    ) -> None:
        self.cross_encoder = cross_encoder or MockCrossEncoder()
        self.input_top_k = input_top_k
        self.output_top_k = output_top_k

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RerankedResult]:
        """Rerank candidates and return top output_top_k."""
        top_candidates = candidates[: self.input_top_k]
        if not top_candidates:
            return []

        doc_texts = [c.document.text for c in top_candidates]
        scores = self.cross_encoder.score(query, doc_texts)

        reranked = [
            RerankedResult(retrieval_result=candidate, rerank_score=score)
            for candidate, score in zip(top_candidates, scores)
        ]
        reranked.sort(key=lambda r: r.rerank_score, reverse=True)
        return reranked[: self.output_top_k]
