"""Cross-Entropy Method (CEM) adversarial trigger optimization — Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from secure_rag_bench.rag.hybrid_retrieval import Document, MockEmbeddingModel
from secure_rag_bench.rag.pipeline import SecureRAGPipeline


@dataclass
class CEMConfig:
    """Configuration for CEM adversarial optimization."""

    prefix_length: int = 10
    num_iterations: int = 30
    num_samples: int = 5000
    elite_fraction: float = 0.2
    smoothing: float = 0.1
    vocab_size: int = 1000
    seed: int = 42


@dataclass
class CEMSample:
    """A single CEM sample with fitness score."""

    prefix_tokens: list[int]
    prefix_text: str
    fitness: float
    malicious_in_top5: bool = False


@dataclass
class CEMResult:
    """Result of CEM optimization run."""

    best_sample: CEMSample
    iterations_run: int
    history: list[float] = field(default_factory=list)
    final_trigger: str = ""
    retrieval_success: bool = False


class CEMEngine:
    """
    Cross-Entropy Method simulation engine (Algorithm 1).

    Maintains uniform distribution over 10-token prefix, samples prefixes,
    prepends to attack_fragment, evaluates cosine similarity to benign query,
    and updates distribution from top-20% elite samples.
    """

    def __init__(
        self,
        config: CEMConfig | None = None,
        embedding_model: MockEmbeddingModel | None = None,
    ) -> None:
        self.config = config or CEMConfig()
        self.embedding_model = embedding_model or MockEmbeddingModel()
        self.rng = np.random.default_rng(self.config.seed)
        self._build_vocab()

    def _build_vocab(self) -> None:
        common_tokens = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "don", "now", "search", "find", "query", "document",
            "report", "data", "information", "summary", "analysis", "review", "update",
            "important", "note", "please", "help", "about", "regarding", "concerning",
            "project", "task", "work", "file", "email", "account", "bank", "transfer",
            "payment", "balance", "security", "policy", "access", "user", "system",
        ]
        while len(common_tokens) < self.config.vocab_size:
            common_tokens.append(f"tok_{len(common_tokens)}")
        self.vocab = common_tokens[: self.config.vocab_size]

    def _init_distribution(self) -> np.ndarray:
        """Uniform distribution over vocabulary for each prefix position."""
        return np.ones((self.config.prefix_length, self.config.vocab_size)) / self.config.vocab_size

    def _sample_prefix(self, distribution: np.ndarray) -> list[int]:
        prefix = []
        for pos in range(self.config.prefix_length):
            token_idx = self.rng.choice(self.config.vocab_size, p=distribution[pos])
            prefix.append(int(token_idx))
        return prefix

    def _prefix_to_text(self, prefix_tokens: list[int]) -> str:
        return " ".join(self.vocab[t] for t in prefix_tokens)

    def _fitness(
        self,
        prefix_text: str,
        attack_fragment: str,
        benign_query: str,
    ) -> float:
        """Evaluate cosine similarity of (prefix + attack) embedding to benign query."""
        combined = f"{prefix_text} {attack_fragment}"
        embeddings = self.embedding_model.embed([combined, benign_query])
        return float(_cosine_similarity(embeddings[0], embeddings[1]))

    def _fitness_batch(
        self,
        prefix_texts: list[str],
        attack_fragment: str,
        benign_query: str,
    ) -> np.ndarray:
        """Score a complete CEM generation with one embedding request."""
        candidates = [f"{prefix} {attack_fragment}" for prefix in prefix_texts]
        embeddings = self.embedding_model.embed([benign_query, *candidates])
        query = embeddings[0]
        candidate_vectors = embeddings[1:]
        return np.array([_cosine_similarity(vector, query) for vector in candidate_vectors])

    def _evaluate_ranking(
        self,
        prefix_text: str,
        attack_fragment: str,
        benign_query: str,
        corpus: list[Document],
        malicious_doc_id: str,
    ) -> bool:
        """Check if malicious document ranks in top-5 after trigger prepend."""
        poisoned_text = f"{prefix_text} {attack_fragment}"
        poisoned_doc = Document(
            doc_id=malicious_doc_id,
            text=poisoned_text,
            metadata={"title": "Poisoned Document", "adversarial": "true"},
        )
        modified_corpus = [d for d in corpus if d.doc_id != malicious_doc_id] + [poisoned_doc]
        pipeline = SecureRAGPipeline(modified_corpus)
        reranked = pipeline.retrieve_only(benign_query)
        top5_ids = [r.retrieval_result.document.doc_id for r in reranked[:5]]
        return malicious_doc_id in top5_ids

    def run(
        self,
        attack_fragment: str,
        benign_query: str,
        corpus: list[Document] | None = None,
        malicious_doc_id: str = "malicious_doc",
    ) -> CEMResult:
        """
        Run CEM optimization to find trigger prefix.

        Algorithm 1:
        - Initialize uniform distribution over 10-token prefix
        - For 30 iterations, sample 5000 prefixes
        - Prepend to attack_fragment, score by cosine similarity to benign query
        - Top-20% elite samples update token distribution
        """
        distribution = self._init_distribution()
        best_sample = CEMSample(prefix_tokens=[], prefix_text="", fitness=-1.0)
        history: list[float] = []
        elite_count = max(1, int(self.config.num_samples * self.config.elite_fraction))

        for iteration in range(self.config.num_iterations):
            prefixes = [self._sample_prefix(distribution) for _ in range(self.config.num_samples)]
            prefix_texts = [self._prefix_to_text(prefix) for prefix in prefixes]
            fitnesses = self._fitness_batch(prefix_texts, attack_fragment, benign_query)
            samples = [
                CEMSample(prefix_tokens=prefix, prefix_text=text, fitness=float(fitness))
                for prefix, text, fitness in zip(prefixes, prefix_texts, fitnesses)
            ]

            for sample in samples:
                if sample.fitness > best_sample.fitness:
                    best_sample = sample

            samples.sort(key=lambda s: s.fitness, reverse=True)
            elite = samples[:elite_count]
            history.append(elite[0].fitness)

            distribution = self._update_distribution(distribution, elite)

        trigger = f"{best_sample.prefix_text} {attack_fragment}"
        if corpus is not None:
            best_sample.malicious_in_top5 = self._evaluate_ranking(
                best_sample.prefix_text,
                attack_fragment,
                benign_query,
                corpus,
                malicious_doc_id,
            )
        return CEMResult(
            best_sample=best_sample,
            iterations_run=self.config.num_iterations,
            history=history,
            final_trigger=trigger.strip(),
            retrieval_success=best_sample.malicious_in_top5,
        )

    def _update_distribution(
        self,
        distribution: np.ndarray,
        elite: list[CEMSample],
    ) -> np.ndarray:
        """Update token distribution from elite samples with smoothing."""
        new_dist = np.zeros_like(distribution)
        for sample in elite:
            for pos, token_idx in enumerate(sample.prefix_tokens):
                new_dist[pos, token_idx] += 1.0

        for pos in range(self.config.prefix_length):
            row = new_dist[pos]
            if row.sum() > 0:
                row = row / row.sum()
            else:
                row = np.ones(self.config.vocab_size) / self.config.vocab_size
            distribution[pos] = (
                (1 - self.config.smoothing) * row
                + self.config.smoothing / self.config.vocab_size
            )

        return distribution


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
