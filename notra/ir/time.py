"""Time signature model for measure attributes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class TimeSignature:
    """Simple time signature represented as beats / beat-type."""

    beats: int
    beat_type: int

    def __post_init__(self) -> None:
        if self.beats <= 0:
            raise ValueError("beats must be > 0")
        if self.beat_type <= 0:
            raise ValueError("beat_type must be > 0")

    @property
    def measure_duration(self) -> Fraction:
        """Expected one-measure duration as fraction of whole note."""
        return Fraction(self.beats, self.beat_type)

    def to_dict(self) -> dict[str, int]:
        """Serialize to a JSON-friendly dictionary."""
        return {"beats": self.beats, "beat_type": self.beat_type}
