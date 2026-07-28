"""Data provenance tracking for CaMeL variables."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Source(str, Enum):
    """Origin of a data value."""

    USER = "user"
    TOOL = "tool"
    DERIVED = "derived"


class Capability(str, Enum):
    """Reader capabilities controlling data access."""

    USER = "user"
    INTERNAL = "internal"
    EXTERNAL = "external"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class Provenance:
    """Provenance metadata attached to every tracked variable."""

    sources: frozenset[Source]
    allowed_readers: frozenset[Capability] = field(
        default_factory=lambda: frozenset({Capability.USER, Capability.INTERNAL})
    )

    def merge(self, other: Provenance) -> Provenance:
        """Merge operand labels without allowing capabilities to expand."""
        sources = self.sources | other.sources
        readers = self.allowed_readers & other.allowed_readers
        if not readers:
            readers = frozenset({Capability.INTERNAL})
        return Provenance(sources=sources, allowed_readers=readers)

    @property
    def source(self) -> Source:
        """Compatibility view for callers that only distinguish tainted data."""
        return Source.TOOL if self.is_tainted() else Source.USER

    def with_readers(self, readers: frozenset[Capability]) -> Provenance:
        return Provenance(sources=self.sources, allowed_readers=readers)

    def is_tainted(self) -> bool:
        return Source.TOOL in self.sources

    def is_untrusted(self) -> bool:
        return self.is_tainted()

    def is_externally_readable(self) -> bool:
        return Capability.EXTERNAL in self.allowed_readers

    def can_send_to(self, recipient: Capability) -> bool:
        return recipient in self.allowed_readers


@dataclass
class TrackedValue(Generic[T]):
    """A value wrapped with provenance metadata."""

    value: T
    provenance: Provenance

    @classmethod
    def from_user(cls, value: T, readers: frozenset[Capability] | None = None) -> TrackedValue[T]:
        default_readers = frozenset({Capability.USER, Capability.INTERNAL, Capability.EXTERNAL})
        return cls(
            value=value,
            provenance=Provenance(
                sources=frozenset({Source.USER}),
                allowed_readers=readers or default_readers,
            ),
        )

    @classmethod
    def from_tool(
        cls,
        value: T,
        readers: frozenset[Capability] | None = None,
    ) -> TrackedValue[T]:
        default_readers = frozenset({Capability.INTERNAL, Capability.QUARANTINED})
        return cls(
            value=value,
            provenance=Provenance(
                sources=frozenset({Source.TOOL}),
                allowed_readers=readers or default_readers,
            ),
        )

    def merge_provenance(self, other: TrackedValue[Any]) -> Provenance:
        return self.provenance.merge(other.provenance)
