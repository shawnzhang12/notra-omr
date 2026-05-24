"""Provenance metadata for traceable IR nodes and artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from notra.core.geometry import BBox


@dataclass(frozen=True, slots=True)
class Provenance:
    """Trace information for where a symbol/claim came from."""

    source: str
    producer: str
    page: int | None = None
    bbox: BBox | None = None
    confidence: float | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if not self.producer.strip():
            raise ValueError("producer must be non-empty")
        if self.page is not None and self.page < 1:
            raise ValueError("page must be >= 1 when provided")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")

    def to_dict(self) -> dict[str, object]:
        """Serialize provenance to a JSON-friendly dictionary."""
        payload: dict[str, object] = {
            "source": self.source,
            "producer": self.producer,
        }
        if self.page is not None:
            payload["page"] = self.page
        if self.bbox is not None:
            payload["bbox"] = self.bbox.to_dict()
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload
