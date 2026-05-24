"""Token export wrapper for Notra IR."""

from __future__ import annotations

from notra.ir.score import Score
from notra.ir.tokens import linearize


def export_score_tokens(score: Score) -> list[str]:
    """Return token sequence for debugging and eval baselines."""
    return linearize(score)
