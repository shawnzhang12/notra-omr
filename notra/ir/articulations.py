"""Articulation constants and helpers for Notra IR."""

from __future__ import annotations

VALID_ARTICULATIONS: frozenset[str] = frozenset(
    {
        "accent",
        "strong-accent",
        "staccato",
        "staccatissimo",
        "tenuto",
        "detached-legato",
        "spiccato",
        "breath-mark",
        "caesura",
        "stress",
        "unstress",
        "soft-accent",
    }
)


def is_valid_articulation(value: str) -> bool:
    """Return True when value is a supported articulation token."""
    return value in VALID_ARTICULATIONS
