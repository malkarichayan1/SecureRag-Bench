# Retrieval and Trigger Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a deterministic hybrid-RAG experiment and CEM trigger-fragment simulator that measures retrieval manipulation without invoking external models.

**Architecture:** Keep injectable embedding and reranker protocols, with deterministic mocks as defaults. The pipeline retrieves top-20, reranks top-5, and emits only an XML-delimited bundle to the quarantined parser. The CEM engine optimizes prefix similarity in the configured embedding space, then evaluates the best candidate against the same retrieval configuration.

**Tech Stack:** Python, NumPy, rank-bm25, pytest; no network access.

## Global Constraints

- Defaults are deterministic and seeded.
- No OpenAI, sentence-transformer, or network call is permitted in tests.
- CEM defaults retain the requested Algorithm 1 settings: 10 tokens, 30 cycles, 5,000 samples, and 20% elite selection.
- Small test configs must exercise identical logic with fewer samples.

### Task 1: Lock down hybrid retrieval and reranking

**Files:**
- Modify: `src/secure_rag_bench/rag/hybrid_retrieval.py`
- Modify: `src/secure_rag_bench/rag/pipeline.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing deterministic ranking tests**

```python
def test_hybrid_retrieval_returns_requested_number_of_descending_results(deterministic_corpus):
    results = HybridRetriever(deterministic_corpus).retrieve("project status", top_k=2)
    assert len(results) == 2
    assert results[0].combined_score >= results[1].combined_score
```

- [ ] **Step 2: Write failing pipeline-boundary tests**

```python
def test_pipeline_sends_only_xml_delimited_content_to_quarantined_parser(deterministic_corpus):
    result = SecureRAGPipeline(deterministic_corpus).run("project status")
    assert result.untrusted_bundle.startswith("<untrusted_content>")
    assert "<document" in result.untrusted_bundle
```

- [ ] **Step 3: Run the retrieval tests and verify the expected boundary or ranking gap**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_retrieval.py -v`

- [ ] **Step 4: Implement only the deterministic fixes required by the tests**

Validate non-negative weights, deterministic tie ordering by document ID, injectable embedding/reranker dependencies in `SecureRAGPipeline`, and XML bundle construction through `wrap_untrusted`.

- [ ] **Step 5: Re-run the retrieval suite**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_retrieval.py -v`

### Task 2: Make CEM optimize and report retrieval success coherently

**Files:**
- Modify: `src/secure_rag_bench/adversarial/cem_engine.py`
- Create: `tests/test_cem_engine.py`

- [ ] **Step 1: Write failing small-config CEM tests**

```python
def test_cem_is_seed_deterministic_and_records_every_iteration():
    config = CEMConfig(prefix_length=3, num_iterations=3, num_samples=20, elite_fraction=0.2, vocab_size=20, seed=7)
    first = CEMEngine(config).run("malicious instruction", "project status")
    second = CEMEngine(config).run("malicious instruction", "project status")
    assert first.final_trigger == second.final_trigger
    assert len(first.history) == 3
```

- [ ] **Step 2: Write a failing ranking-evaluation test**

```python
def test_cem_marks_the_best_sample_when_its_poisoned_document_enters_top_five(deterministic_corpus):
    result = CEMEngine(CEMConfig(prefix_length=2, num_iterations=2, num_samples=10, vocab_size=20)).run(
        "project status", "project status", deterministic_corpus, malicious_doc_id="injection-001"
    )
    assert isinstance(result.best_sample.malicious_in_top5, bool)
```

- [ ] **Step 3: Run CEM tests to expose the current per-sample full-pipeline inefficiency or result-reporting gap**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_cem_engine.py -v`

- [ ] **Step 4: Implement deterministic CEM reporting**

Use the injected embedding model consistently for fitness and ranking. Track whether the final best prefix reaches top-5, add a `retrieval_success` field to `CEMResult`, validate config bounds, and preserve smoothed categorical updates.

- [ ] **Step 5: Re-run CEM tests**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_cem_engine.py -v`

### Task 3: Verify the retrieval experiment

**Files:**
- Modify: `README.md`
- Test: `tests/test_retrieval.py`
- Test: `tests/test_cem_engine.py`

- [ ] **Step 1: Document the experiment verification command**

Add `.venv\\Scripts\\python.exe -m pytest tests/test_retrieval.py tests/test_cem_engine.py -v` to the README.

- [ ] **Step 2: Run focused verification**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_retrieval.py tests/test_cem_engine.py -v`

- [ ] **Step 3: Run full verification**

Run: `.venv\\Scripts\\python.exe -m pytest -v`

## Plan self-review

This plan covers deterministic hybrid retrieval, top-20 to top-5 reranking, XML quarantine boundaries, CEM seed reproducibility, Algorithm 1 configuration, and top-5 attack reporting. Benchmark orchestration remains the next milestone.
