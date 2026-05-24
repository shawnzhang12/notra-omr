"""Unit tests for broader IR validation behavior."""

from __future__ import annotations

from typing import Iterable

from notra.ir.measure import Measure, MeasureAttributes, Voice
from notra.ir.note import Duration, Note, Pitch
from notra.ir.score import Part, Score
from notra.ir.time import TimeSignature
from notra.ir.validate import validate_score


def _make_measure(
    *,
    measure_id: str,
    number: int,
    events: Iterable[Note],
    voice_id: str = "v1",
    include_time: bool = False,
) -> Measure:
    attributes = (
        MeasureAttributes(time=TimeSignature(beats=4, beat_type=4), divisions=4)
        if include_time
        else None
    )
    return Measure(
        id=measure_id,
        number=number,
        voices=[Voice(id=voice_id, events=list(events))],
        attributes=attributes,
    )


def _score_with_measures(*measures: Measure) -> Score:
    return Score(
        id="score-001",
        title="Validation Cases",
        parts=[Part(id="P1", name="Piano", measures=list(measures))],
    )


def test_missing_time_signature_is_reported() -> None:
    score = _score_with_measures(
        _make_measure(
            measure_id="measure-001",
            number=1,
            events=[
                Note(
                    id="event-001",
                    pitch=Pitch(step="C", octave=4),
                    duration=Duration(1, 4),
                )
            ],
        )
    )

    report = validate_score(score)
    codes = {issue.code for issue in report.issues}

    assert report.has_errors
    assert "MISSING_TIME_SIGNATURE" in codes


def test_duplicate_event_id_is_reported() -> None:
    m1 = _make_measure(
        measure_id="measure-001",
        number=1,
        include_time=True,
        events=[
            Note(id="event-001", pitch=Pitch(step="C", octave=4), duration=Duration(1, 1), voice=1)
        ],
    )
    m2 = _make_measure(
        measure_id="measure-002",
        number=2,
        events=[
            Note(id="event-001", pitch=Pitch(step="D", octave=4), duration=Duration(1, 1), voice=1)
        ],
    )

    report = validate_score(_score_with_measures(m1, m2))
    codes = {issue.code for issue in report.issues}

    assert "DUPLICATE_EVENT_ID" in codes


def test_tie_stop_without_start_is_reported() -> None:
    measure = _make_measure(
        measure_id="measure-001",
        number=1,
        include_time=True,
        events=[
            Note(
                id="event-001",
                pitch=Pitch(step="C", octave=4),
                duration=Duration(1, 1),
                ties=("stop",),
                voice=1,
            )
        ],
    )

    report = validate_score(_score_with_measures(measure))
    codes = {issue.code for issue in report.issues}

    assert "TIE_STOP_WITHOUT_START" in codes


def test_unclosed_tie_at_part_end_is_reported() -> None:
    measure = _make_measure(
        measure_id="measure-001",
        number=1,
        include_time=True,
        events=[
            Note(
                id="event-001",
                pitch=Pitch(step="C", octave=4),
                duration=Duration(1, 1),
                ties=("start",),
                voice=1,
            )
        ],
    )

    report = validate_score(_score_with_measures(measure))
    codes = {issue.code for issue in report.issues}

    assert "UNCLOSED_TIE_AT_PART_END" in codes


def test_event_voice_mismatch_is_warning() -> None:
    measure = _make_measure(
        measure_id="measure-001",
        number=1,
        include_time=True,
        voice_id="v2",
        events=[
            Note(id="event-001", pitch=Pitch(step="C", octave=4), duration=Duration(1, 1), voice=1)
        ],
    )

    report = validate_score(_score_with_measures(measure))
    mismatch_issues = [issue for issue in report.issues if issue.code == "EVENT_VOICE_MISMATCH"]

    assert mismatch_issues
    assert mismatch_issues[0].severity.value == "warning"


def test_non_monotonic_measure_order_is_warning() -> None:
    m1 = _make_measure(
        measure_id="measure-010",
        number=10,
        include_time=True,
        events=[
            Note(id="event-010", pitch=Pitch(step="C", octave=4), duration=Duration(1, 1), voice=1)
        ],
    )
    m2 = _make_measure(
        measure_id="measure-009",
        number=9,
        events=[
            Note(id="event-011", pitch=Pitch(step="D", octave=4), duration=Duration(1, 1), voice=1)
        ],
    )

    report = validate_score(_score_with_measures(m1, m2))
    codes = {issue.code for issue in report.issues}

    assert "NON_MONOTONIC_MEASURE_ORDER" in codes
