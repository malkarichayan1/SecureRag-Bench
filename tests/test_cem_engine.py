from secure_rag_bench.adversarial.cem_engine import CEMConfig, CEMEngine
from secure_rag_bench.rag.hybrid_retrieval import MockEmbeddingModel


def test_cem_is_seed_deterministic_and_records_every_iteration() -> None:
    config = CEMConfig(
        prefix_length=3,
        num_iterations=3,
        num_samples=20,
        elite_fraction=0.2,
        vocab_size=20,
        seed=7,
    )

    first = CEMEngine(config).run("malicious instruction", "project status")
    second = CEMEngine(config).run("malicious instruction", "project status")

    assert first.final_trigger == second.final_trigger
    assert len(first.history) == 3


def test_cem_reports_top_five_retrieval_success(deterministic_corpus) -> None:
    result = CEMEngine(
        CEMConfig(prefix_length=2, num_iterations=2, num_samples=10, vocab_size=20, seed=3)
    ).run(
        "project status",
        "project status",
        deterministic_corpus,
        malicious_doc_id="injection-001",
    )

    assert isinstance(result.retrieval_success, bool)
    assert result.retrieval_success == result.best_sample.malicious_in_top5


def test_cem_batches_embedding_work_per_iteration() -> None:
    class CountingEmbeddingModel(MockEmbeddingModel):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def embed(self, texts: list[str]):
            self.calls += 1
            return super().embed(texts)

    embedding_model = CountingEmbeddingModel()
    CEMEngine(
        CEMConfig(prefix_length=2, num_iterations=3, num_samples=20, vocab_size=20, seed=5),
        embedding_model=embedding_model,
    ).run("malicious instruction", "project status")

    assert embedding_model.calls <= 4
