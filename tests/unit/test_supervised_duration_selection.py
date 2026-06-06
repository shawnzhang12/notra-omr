"""Tests for supervised duration-sequence candidate selection."""

from __future__ import annotations

from notra.semantics import DurationCandidate, SymbolCandidate
from notra.semantics.supervised_duration import select_candidates_for_duration_sequence


def test_supervised_selector_matches_target_count_and_order() -> None:
    candidates = [
        _candidate("noise", 5.0, 480, score=-2.0),
        _candidate("a", 10.0, 480),
        _candidate("b", 30.0, 960),
        _candidate("extra", 32.0, 480, score=-2.0),
        _candidate("c", 60.0, 480),
    ]

    selection = select_candidates_for_duration_sequence(
        candidates,
        (480, 960, 480),
        expected_ticks=1920,
    )

    assert selection.valid
    assert selection.selected_candidate_ids == ("a", "b", "c")
    assert selection.selected_count == 3


def test_supervised_selector_reports_candidate_shortfall() -> None:
    selection = select_candidates_for_duration_sequence(
        [_candidate("a", 10.0, 480)],
        (480, 480),
        expected_ticks=960,
    )

    assert not selection.valid
    assert selection.selected_candidate_ids == ()
    assert selection.diagnostics == ("candidate shortfall: 1 < 2",)


def _candidate(
    candidate_id: str,
    x: float,
    duration_ticks: int,
    *,
    score: float = 0.0,
) -> SymbolCandidate:
    return SymbolCandidate(
        id=candidate_id,
        bbox=(0, 0, 10, 10),
        staff_id=0,
        x=x,
        y=0.0,
        duration_candidates=[
            DurationCandidate(
                duration_ticks=duration_ticks,
                note_type="quarter",
                visual_score=score,
            )
        ],
    )
