"""Key signature model for measure attributes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KeySignature:
    """Circle-of-fifths key signature."""

    fifths: int
    mode: str = "major"

    def __post_init__(self) -> None:
        if self.mode not in {"major", "minor"}:
            raise ValueError("mode must be 'major' or 'minor'")
        if not (-7 <= self.fifths <= 7):
            raise ValueError("fifths must be in [-7, 7]")

    def to_dict(self) -> dict[str, int | str]:
        """Serialize to a JSON-friendly dictionary."""
        return {"fifths": self.fifths, "mode": self.mode}
