"""Barline model for measure boundaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Barline:
    """Barline style descriptor."""

    style: str = "regular"

    def __post_init__(self) -> None:
        valid = {"regular", "double", "final", "repeat-left", "repeat-right"}
        if self.style not in valid:
            raise ValueError(f"invalid barline style: {self.style!r}")

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-friendly dictionary."""
        return {"style": self.style}
