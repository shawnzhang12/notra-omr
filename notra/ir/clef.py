"""Clef model for measure attributes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Clef:
    """Staff clef configuration."""

    sign: str
    line: int

    def __post_init__(self) -> None:
        if self.sign not in {"G", "F", "C"}:
            raise ValueError(f"invalid clef sign: {self.sign!r}")
        if not (1 <= self.line <= 5):
            raise ValueError("clef line must be in [1, 5]")

    def to_dict(self) -> dict[str, int | str]:
        """Serialize to a JSON-friendly dictionary."""
        return {"sign": self.sign, "line": self.line}
