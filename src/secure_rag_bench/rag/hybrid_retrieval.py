"""Hybrid retrieval: semantic (embeddings) + BM25 keyword search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Document:
    """A document in the retrieval corpus."""

    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A scored retrieval result."""

    document: Document
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for text embedding models."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return embeddings array of shape (n, dim)."""
        ...


class MockEmbeddingModel:
    """Deterministic hash-based embeddings for tests (no API required)."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            codepoints = np.frombuffer(
                text.lower().encode("utf-32-le"), dtype="<u4"
            ).astype(float)
            positions = np.arange(codepoints.size) % self.dim
            vec = np.bincount(
                positions,
                weights=codepoints / 256.0,
                minlength=self.dim,
            ).astype(float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors)


class OpenAIEmbeddingModel:
    """OpenAI text-embedding-3-small integration (optional)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        import os

        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY required")
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.model, input=texts)
        return np.array([item.embedding for item in response.data])


class HybridRetriever:
    """Combines semantic and BM25 retrieval with score fusion."""

    def __init__(
        self,
        documents: list[Document],
        embedding_model: EmbeddingModel | None = None,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> None:
        if semantic_weight < 0 or bm25_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        if not np.isclose(semantic_weight + bm25_weight, 1.0):
            raise ValueError("retrieval weights must sum to 1.0")
        self.documents = documents
        self.embedding_model = embedding_model or MockEmbeddingModel()
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self._doc_embeddings: np.ndarray | None = None
        self._bm25 = None
        self._index()

    def _index(self) -> None:
        texts = [doc.text for doc in self.documents]
        self._doc_embeddings = self.embedding_model.embed(texts)

        try:
            from rank_bm25 import BM25Okapi

            tokenized = [text.lower().split() for text in texts]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """Retrieve top-k documents using hybrid scoring."""
        if not self.documents:
            return []

        query_embedding = self.embedding_model.embed([query])[0]
        semantic_scores = _cosine_similarity_batch(query_embedding, self._doc_embeddings)

        if self._bm25 is not None:
            bm25_scores = np.array(self._bm25.get_scores(query.lower().split()))
            bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
            bm25_norm = bm25_scores / bm25_max
        else:
            bm25_norm = _keyword_overlap_scores(query, self.documents)

        results: list[RetrievalResult] = []
        for i, doc in enumerate(self.documents):
            combined = self.semantic_weight * semantic_scores[i] + self.bm25_weight * bm25_norm[i]
            results.append(
                RetrievalResult(
                    document=doc,
                    semantic_score=float(semantic_scores[i]),
                    bm25_score=float(bm25_norm[i]),
                    combined_score=float(combined),
                )
            )

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]


def _cosine_similarity_batch(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    if doc_matrix is None or len(doc_matrix) == 0:
        return np.array([])
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm


def _keyword_overlap_scores(query: str, documents: list[Document]) -> np.ndarray:
    query_tokens = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_tokens = set(doc.text.lower().split())
        overlap = len(query_tokens & doc_tokens)
        scores.append(overlap / max(len(query_tokens), 1))
    return np.array(scores)
