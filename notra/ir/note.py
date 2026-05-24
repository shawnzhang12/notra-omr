"""Pitch, duration, and note event models for Notra IR."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from notra.core.provenance import Provenance

_VALID_STEPS = {"A", "B", "C", "D", "E", "F", "G"}
_VALID_TIES = {"start", "stop", "continue"}


@dataclass(frozen=True, slots=True)
class Duration:
    """Rational duration relative to a whole note."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator <= 0:
            raise ValueError("numerator must be > 0")
        if self.denominator <= 0:
            raise ValueError("denominator must be > 0")

    @property
    def fraction(self) -> Fraction:
        """Duration as an exact fraction of a whole note."""
        return Fraction(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, int]:
        """Serialize to a JSON-friendly dictionary."""
        return {"numerator": self.numerator, "denominator": self.denominator}


@dataclass(frozen=True, slots=True)
class Pitch:
    """Written pitch representation preserving spelling."""

    step: str
    octave: int
    alter: int = 0

    def __post_init__(self) -> None:
        if self.step not in _VALID_STEPS:
            raise ValueError(f"invalid step: {self.step!r}")
        if not (0 <= self.octave <= 9):
            raise ValueError("octave must be in [0, 9]")
        if not (-2 <= self.alter <= 2):
            raise ValueError("alter must be in [-2, 2]")

    def to_dict(self) -> dict[str, int | str]:
        """Serialize to a JSON-friendly dictionary."""
        payload: dict[str, int | str] = {"step": self.step, "octave": self.octave}
        if self.alter != 0:
            payload["alter"] = self.alter
        return payload


@dataclass(frozen=True, slots=True)
class Note:
    """One pitched event in a voice."""

    id: str
    pitch: Pitch
    duration: Duration
    voice: int = 1
    ties: tuple[str, ...] = ()
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("id must be non-empty")
        if self.voice < 1:
            raise ValueError("voice must be >= 1")
        for tie in self.ties:
            if tie not in _VALID_TIES:
                raise ValueError(f"invalid tie: {tie!r}")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        payload: dict[str, object] = {
            "kind": "note",
            "id": self.id,
            "pitch": self.pitch.to_dict(),
            "duration": self.duration.to_dict(),
            "voice": self.voice,
        }
        if self.ties:
            payload["ties"] = list(self.ties)
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload
