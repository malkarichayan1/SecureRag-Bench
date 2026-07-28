"""Privileged LLM: generates Python code expressing task intent from user queries."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class PrivilegedLLM(Protocol):
    """Protocol for the privileged model that plans tasks as Python code."""

    def generate_plan(self, user_query: str, *, context: str | None = None) -> str:
        """Generate Python code expressing the user's task intent."""
        ...


class MockPrivilegedLLM:
    """Deterministic privileged LLM for tests and offline demos."""

    def generate_plan(self, user_query: str, *, context: str | None = None) -> str:
        query_lower = user_query.lower()
        if "email" in query_lower or "send" in query_lower:
            return (
                'recipient = "team@company.com"\n'
                'subject = "Update"\n'
                "body = user_input\n"
                "send_email(to=recipient, subject=subject, body=body)\n"
            )
        if "transfer" in query_lower:
            return 'destination = "savings account"\ntransfer(to=destination, amount=50)\n'
        return (
            'query = "search documents"\n'
            "results = retrieve(query=query, top_k=5)\n"
            "parsed = quarantine_parse(results, schema='DocumentSummary')\n"
            "answer = format_response(parsed)\n"
        )


class OpenAIPrivilegedLLM(ABC):
    """OpenAI-backed privileged LLM (optional, requires API key)."""

    SYSTEM_PROMPT = """You are a privileged planning model. Generate ONLY valid Python code
that expresses the user's task intent. Rules:
- NO while loops
- NO import statements
- NO eval or exec
- NO list.append(); use [*list, elem] instead
- Use provided tool functions: retrieve, quarantine_parse, format_response, send_email
- Assign final result to variable 'answer' when applicable
"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def generate_plan(self, user_query: str, *, context: str | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY required for OpenAIPrivilegedLLM")

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": user_query})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        return _extract_code_block(content)


def _extract_code_block(text: str) -> str:
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    return text.strip()
