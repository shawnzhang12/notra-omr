"""Pitch, duration, and note event models for Notra IR."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from notra.core.provenance import Provenance
from notra.ir.articulations import is_valid_articulation

_VALID_STEPS = {"A", "B", "C", "D", "E", "F", "G"}
_VALID_TIES = {"start", "stop", "continue"}
_VALID_SPANNERS = {"start", "stop", "continue"}
_VALID_BEAM_VALUES = {"begin", "continue", "end", "forward hook", "backward hook"}
_VALID_TUPLET = {"start", "stop"}


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
    slurs: tuple[str, ...] = ()
    articulations: tuple[str, ...] = ()
    beams: tuple[str, ...] = ()
    lyric: str | None = None
    fingering: str | None = None
    chord: bool = False
    tuplet: str | None = None
    tuplet_ratio: tuple[int, int] | None = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("id must be non-empty")
        if self.voice < 1:
            raise ValueError("voice must be >= 1")
        for tie in self.ties:
            if tie not in _VALID_TIES:
                raise ValueError(f"invalid tie: {tie!r}")
        for slur in self.slurs:
            if slur not in _VALID_SPANNERS:
                raise ValueError(f"invalid slur marker: {slur!r}")
        for articulation in self.articulations:
            if not is_valid_articulation(articulation):
                raise ValueError(f"invalid articulation: {articulation!r}")
        for beam in self.beams:
            if beam not in _VALID_BEAM_VALUES:
                raise ValueError(f"invalid beam value: {beam!r}")
        if self.lyric is not None and not self.lyric.strip():
            raise ValueError("lyric must be non-empty when provided")
        if self.fingering is not None and not self.fingering.strip():
            raise ValueError("fingering must be non-empty when provided")
        if self.tuplet is not None and self.tuplet not in _VALID_TUPLET:
            raise ValueError("tuplet must be one of: start, stop")
        if self.tuplet_ratio is not None:
            actual, normal = self.tuplet_ratio
            if actual <= 0 or normal <= 0:
                raise ValueError("tuplet_ratio values must be > 0")

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
        if self.slurs:
            payload["slurs"] = list(self.slurs)
        if self.articulations:
            payload["articulations"] = list(self.articulations)
        if self.beams:
            payload["beams"] = list(self.beams)
        if self.lyric is not None:
            payload["lyric"] = self.lyric
        if self.fingering is not None:
            payload["fingering"] = self.fingering
        if self.chord:
            payload["chord"] = True
        if self.tuplet is not None:
            payload["tuplet"] = self.tuplet
        if self.tuplet_ratio is not None:
            payload["tuplet_ratio"] = {
                "actual_notes": self.tuplet_ratio[0],
                "normal_notes": self.tuplet_ratio[1],
            }
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload
