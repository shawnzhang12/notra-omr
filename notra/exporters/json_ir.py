"""JSON export wrapper for Notra score IR."""

from __future__ import annotations

from notra.ir.score import Score
from notra.ir.serialize import score_to_json


def export_score_to_json(score: Score) -> str:
    """Return canonical JSON serialization for a score."""
    return score_to_json(score, indent=2)
