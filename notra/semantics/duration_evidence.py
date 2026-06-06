"""Rule-based duration evidence for measure-constrained decoding.

This layer intentionally produces hypotheses, not final labels. Visual evidence
is local and fallible; the rhythm solver remains responsible for choosing a
measure-valid assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from notra.semantics import DurationCandidate, generate_duration_candidates


@dataclass(frozen=True, slots=True)
class DurationEvidence:
    """Local visual evidence used to build duration hypotheses."""

    is_filled: bool
    has_stem: bool
    flag_count: int = 0
    dot_count: int = 0
    is_rest: bool = False


def generate_duration_candidates_from_evidence(
    evidence: DurationEvidence,
) -> list[DurationCandidate]:
    """Convert local evidence into ranked duration candidates."""
    return generate_duration_candidates(
        is_filled=evidence.is_filled,
        has_stem=evidence.has_stem,
        flag_count=evidence.flag_count,
        dot_count=evidence.dot_count,
        is_rest=evidence.is_rest,
    )
