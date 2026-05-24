"""Unit tests for measure duration validation."""

from __future__ import annotations

from notra.ir.barline import Barline
from notra.ir.clef import Clef
from notra.ir.key import KeySignature
from notra.ir.measure import Measure, MeasureAttributes, Voice
from notra.ir.note import Duration, Note, Pitch
from notra.ir.rest import Rest
from notra.ir.score import Part, Score
from notra.ir.time import TimeSignature
from notra.ir.validate import validate_score


def _build_score_with_voice_events(note_count: int) -> Score:
    events: list[Note | Rest] = [
        Note(
            id=f"event-{index + 1:03d}",
            pitch=Pitch(step="C", octave=4),
            duration=Duration(1, 4),
            voice=1,
        )
        for index in range(note_count)
    ]
    measure = Measure(
        id="measure-001",
        number=1,
        voices=[Voice(id="v1", events=events)],
        attributes=MeasureAttributes(
            clef=Clef(sign="G", line=2),
            key=KeySignature(fifths=0, mode="major"),
            time=TimeSignature(beats=4, beat_type=4),
            divisions=4,
        ),
        barline=Barline(style="regular"),
    )
    return Score(
        id="score-001",
        title="Validation Case",
        parts=[Part(id="P1", name="Piano", measures=[measure])],
    )


def test_measure_validation_passes_when_balanced() -> None:
    report = validate_score(_build_score_with_voice_events(4))
    assert not report.has_errors


def test_measure_validation_fails_on_underflow() -> None:
    report = validate_score(_build_score_with_voice_events(3))
    assert report.has_errors
    assert report.issues[0].code == "MEASURE_DURATION_UNDERFLOW"


def test_measure_validation_fails_on_overflow() -> None:
    report = validate_score(_build_score_with_voice_events(5))
    assert report.has_errors
    assert report.issues[0].code == "MEASURE_DURATION_OVERFLOW"
