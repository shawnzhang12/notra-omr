"""Rest event model for Notra IR."""

from __future__ import annotations

from dataclasses import dataclass

from notra.core.provenance import Provenance
from notra.ir.note import Duration


@dataclass(frozen=True, slots=True)
class Rest:
    """One rest event in a voice."""

    id: str
    duration: Duration
    voice: int = 1
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("id must be non-empty")
        if self.voice < 1:
            raise ValueError("voice must be >= 1")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        payload: dict[str, object] = {
            "kind": "rest",
            "id": self.id,
            "duration": self.duration.to_dict(),
            "voice": self.voice,
        }
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload
