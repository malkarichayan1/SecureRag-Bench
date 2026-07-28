from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


@dataclass
class SimulatedToolbox:
    documents: dict[str, str]
    calls: list[ToolCall] = field(default_factory=list)

    def local_search(self, query: str) -> list[str]:
        self.calls.append(ToolCall("local_search", {"query": query}))
        normalized_query = query.lower()
        return [
            text
            for doc_id, text in sorted(self.documents.items())
            if normalized_query in f"{doc_id} {text}".lower()
        ]

    def create_email_draft(self, recipient: str, subject: str, body: str) -> dict[str, str]:
        self.calls.append(
            ToolCall(
                "create_email_draft",
                {"recipient": recipient, "subject": subject, "body": body},
            )
        )
        return {
            "status": "draft",
            "recipient": recipient,
            "subject": subject,
            "body": body,
        }

    def send_email(self, recipient: str, subject: str, body: str) -> dict[str, str]:
        self.calls.append(
            ToolCall(
                "send_email",
                {"recipient": recipient, "subject": subject, "body": body},
            )
        )
        return {
            "status": "sent",
            "recipient": recipient,
            "subject": subject,
            "body": body,
        }
