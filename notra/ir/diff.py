"""Structural comparison helpers for Notra score IR."""

from __future__ import annotations

from notra.ir.score import Score
from notra.ir.serialize import score_to_dict


def structurally_equal(left: Score, right: Score) -> bool:
    """Return True when two scores serialize to the same structure."""
    return score_to_dict(left) == score_to_dict(right)
