"""SecureRAG-Bench: Secure RAG with CaMeL dual-LLM architecture."""

__version__ = "0.1.0"

from secure_rag_bench.camel.interpreter import CaMeLInterpreter
from secure_rag_bench.camel.privileged_llm import PrivilegedLLM, MockPrivilegedLLM
from secure_rag_bench.camel.provenance import Capability, Provenance, Source
from secure_rag_bench.camel.quarantined_llm import MockQuarantinedLLM, QuarantinedLLM
from secure_rag_bench.security.policy import PolicyDecision, check_policy
from secure_rag_bench.rag.pipeline import SecureRAGPipeline

__all__ = [
    "__version__",
    "CaMeLInterpreter",
    "PrivilegedLLM",
    "MockPrivilegedLLM",
    "QuarantinedLLM",
    "MockQuarantinedLLM",
    "Provenance",
    "Source",
    "Capability",
    "check_policy",
    "PolicyDecision",
    "SecureRAGPipeline",
]
