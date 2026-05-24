"""Dynamics constants and helpers for Notra IR."""

from __future__ import annotations

VALID_DYNAMICS: frozenset[str] = frozenset(
    {
        "pppp",
        "ppp",
        "pp",
        "p",
        "mp",
        "mf",
        "f",
        "ff",
        "fff",
        "ffff",
        "sfz",
        "fp",
    }
)


def is_valid_dynamic(value: str) -> bool:
    """Return True when value is a supported dynamic mark."""
    return value in VALID_DYNAMICS
