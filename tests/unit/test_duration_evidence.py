"""Tests for rule-based duration evidence."""

from __future__ import annotations

from types import SimpleNamespace

from notra.core.geometry import BBox
from notra.semantics import (
    EIGHTH,
    HALF,
    QUARTER,
    SIXTEENTH,
    WHOLE,
    generate_duration_candidates,
)
from notra.semantics.duration_evidence import (
    DurationEvidence,
    generate_duration_candidates_from_evidence,
)
from notra.semantics.rhythm_solver import build_candidates_from_events


def test_base_duration_rules_emit_expected_primary_candidates() -> None:
    assert generate_duration_candidates(False, False)[0].duration_ticks == WHOLE
    assert generate_duration_candidates(False, True)[0].duration_ticks == HALF
    assert generate_duration_candidates(True, True)[0].duration_ticks == QUARTER
    assert generate_duration_candidates(True, True, flag_count=1)[0].duration_ticks == EIGHTH
    assert generate_duration_candidates(True, True, flag_count=2)[0].duration_ticks == SIXTEENTH


def test_dotted_duration_evidence_prefers_dotted_candidate() -> None:
    candidates = generate_duration_candidates(
        is_filled=True,
        has_stem=True,
        flag_count=0,
        dot_count=1,
    )

    assert candidates[0].duration_ticks == QUARTER
    assert candidates[0].dots == 1
    assert candidates[0].adjusted_ticks == QUARTER + QUARTER // 2
    assert any(candidate.dots == 0 for candidate in candidates)


def test_flagged_duration_evidence_keeps_unflagged_fallbacks() -> None:
    one_flag = generate_duration_candidates(True, True, flag_count=1)
    two_flags = generate_duration_candidates(True, True, flag_count=2)

    assert one_flag[0].duration_ticks == EIGHTH
    assert {candidate.duration_ticks for candidate in one_flag} >= {EIGHTH, QUARTER}
    assert two_flags[0].duration_ticks == SIXTEENTH
    assert {candidate.duration_ticks for candidate in two_flags} >= {
        SIXTEENTH,
        EIGHTH,
        QUARTER,
    }


def test_duration_evidence_wrapper_preserves_visual_inputs() -> None:
    candidates = generate_duration_candidates_from_evidence(
        DurationEvidence(
            is_filled=True,
            has_stem=True,
            flag_count=2,
            dot_count=1,
        )
    )

    assert candidates[0].duration_ticks == SIXTEENTH
    assert candidates[0].adjusted_ticks == SIXTEENTH + SIXTEENTH // 2
    assert "augmentation_dot" in candidates[0].evidence


def test_single_symbol_measure_context_adds_measure_fill_candidate() -> None:
    event = SimpleNamespace(
        event_index=0,
        cx=50.0,
        cy=40.0,
        bbox=BBox(45, 35, 55, 45),
        staff_index=0,
        is_rest=False,
    )
    notehead = SimpleNamespace(is_filled=False, area=100.0)
    boundary = SimpleNamespace(measure_number=1, x_start=0.0, x_end=100.0)

    per_measure = build_candidates_from_events(
        [event],
        stem_map={},
        flag_map={},
        noteheads=[notehead],
        measure_boundaries=[boundary],
        expected_measure_ticks=1440,
    )

    assert per_measure[0][0].duration_candidates[0].adjusted_ticks == 1440
    assert per_measure[0][0].duration_candidates[0].evidence == "single_symbol_measure_fill"


def test_candidate_builder_keeps_system_measures_separate() -> None:
    events = [
        SimpleNamespace(
            event_index=0,
            cx=50.0,
            cy=40.0,
            bbox=BBox(45, 35, 55, 45),
            staff_index=0,
            is_rest=False,
        ),
        SimpleNamespace(
            event_index=1,
            cx=50.0,
            cy=140.0,
            bbox=BBox(45, 135, 55, 145),
            staff_index=1,
            is_rest=False,
        ),
    ]
    noteheads = [
        SimpleNamespace(is_filled=True, area=100.0),
        SimpleNamespace(is_filled=True, area=100.0),
    ]
    boundaries = [
        SimpleNamespace(measure_number=1, x_start=0.0, x_end=100.0, system_index=0),
        SimpleNamespace(measure_number=1, x_start=0.0, x_end=100.0, system_index=1),
    ]

    per_measure = build_candidates_from_events(
        events,
        stem_map={0: object(), 1: object()},
        flag_map={},
        noteheads=noteheads,
        measure_boundaries=boundaries,
        system_members=[[0], [1]],
    )

    assert len(per_measure) == 2
    assert [measure[0].measure_id for measure in per_measure] == ["s0_m1", "s1_m1"]
    assert [measure[0].id for measure in per_measure] == ["evt_0", "evt_1"]
