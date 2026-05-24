"""Measure-level IR models."""

from __future__ import annotations

from dataclasses import dataclass, field

from notra.ir.barline import Barline
from notra.ir.clef import Clef
from notra.ir.key import KeySignature
from notra.ir.note import Note
from notra.ir.rest import Rest
from notra.ir.time import TimeSignature

Event = Note | Rest


@dataclass(frozen=True, slots=True)
class MeasureAttributes:
    """Notation attributes that can change by measure."""

    clef: Clef | None = None
    key: KeySignature | None = None
    time: TimeSignature | None = None
    divisions: int | None = None

    def __post_init__(self) -> None:
        if self.divisions is not None and self.divisions <= 0:
            raise ValueError("divisions must be > 0 when provided")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        payload: dict[str, object] = {}
        if self.clef is not None:
            payload["clef"] = self.clef.to_dict()
        if self.key is not None:
            payload["key"] = self.key.to_dict()
        if self.time is not None:
            payload["time"] = self.time.to_dict()
        if self.divisions is not None:
            payload["divisions"] = self.divisions
        return payload


@dataclass(frozen=True, slots=True)
class Voice:
    """Ordered stream of events for one measure voice."""

    id: str
    events: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("voice id must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        return {"id": self.id, "events": [event.to_dict() for event in self.events]}


@dataclass(frozen=True, slots=True)
class Measure:
    """One logical measure containing one or more voices."""

    id: str
    number: int
    voices: list[Voice]
    attributes: MeasureAttributes | None = None
    barline: Barline | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("measure id must be non-empty")
        if self.number < 1:
            raise ValueError("measure number must be >= 1")
        if not self.voices:
            raise ValueError("measure must contain at least one voice")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        payload: dict[str, object] = {
            "id": self.id,
            "number": self.number,
            "voices": [voice.to_dict() for voice in self.voices],
        }
        if self.attributes is not None:
            payload["attributes"] = self.attributes.to_dict()
        if self.barline is not None:
            payload["barline"] = self.barline.to_dict()
        return payload
