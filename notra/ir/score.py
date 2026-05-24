"""Top-level score IR models."""

from __future__ import annotations

from dataclasses import dataclass, field

from notra.ir.measure import Measure


@dataclass(frozen=True, slots=True)
class Part:
    """One logical part in a score."""

    id: str
    name: str
    measures: list[Measure] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("part id must be non-empty")
        if not self.name.strip():
            raise ValueError("part name must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "measures": [measure.to_dict() for measure in self.measures],
        }


@dataclass(frozen=True, slots=True)
class Score:
    """Root Notra score model."""

    id: str
    title: str
    parts: list[Part] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("score id must be non-empty")
        if not self.title.strip():
            raise ValueError("score title must be non-empty")
        if not self.parts:
            raise ValueError("score must contain at least one part")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "ir_version": "0.1.0",
            "id": self.id,
            "title": self.title,
            "parts": [part.to_dict() for part in self.parts],
        }
