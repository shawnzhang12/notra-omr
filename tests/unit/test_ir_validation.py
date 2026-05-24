"""Unit tests for broader IR validation behavior."""

from __future__ import annotations

from notra.ir.measure import Measure, Voice
from notra.ir.note import Duration, Note, Pitch
from notra.ir.score import Part, Score
from notra.ir.validate import validate_score


def test_missing_time_signature_is_reported() -> None:
    score = Score(
        id="score-001",
        title="Missing Time",
        parts=[
            Part(
                id="P1",
                name="Piano",
                measures=[
                    Measure(
                        id="measure-001",
                        number=1,
                        voices=[
                            Voice(
                                id="v1",
                                events=[
                                    Note(
                                        id="event-001",
                                        pitch=Pitch(step="C", octave=4),
                                        duration=Duration(1, 4),
                                    )
                                ],
                            )
                        ],
                    )
                ],
            )
        ],
    )

    report = validate_score(score)
    assert report.has_errors
    assert report.issues[0].code == "MISSING_TIME_SIGNATURE"
