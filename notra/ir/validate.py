"""Semantic validation for Notra score IR."""

from __future__ import annotations

from fractions import Fraction

from notra.core.errors import Severity, ValidationIssue, ValidationReport
from notra.ir.measure import Measure
from notra.ir.score import Part, Score
from notra.ir.time import TimeSignature


def validate_score(score: Score) -> ValidationReport:
    """Validate score-level semantic constraints and return a report."""
    report = ValidationReport()
    for part in score.parts:
        _validate_part(part, report)
    return report


def _validate_part(part: Part, report: ValidationReport) -> None:
    current_time: TimeSignature | None = None
    seen_measure_numbers: set[int] = set()

    for measure in part.measures:
        if measure.number in seen_measure_numbers:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="DUPLICATE_MEASURE_NUMBER",
                    message=f"Duplicate measure number {measure.number} in part {part.id}.",
                    node_id=measure.id,
                )
            )
        seen_measure_numbers.add(measure.number)

        if measure.attributes and measure.attributes.time is not None:
            current_time = measure.attributes.time

        _validate_measure(measure, report, current_time)


def _validate_measure(
    measure: Measure,
    report: ValidationReport,
    current_time: TimeSignature | None,
) -> None:
    if current_time is None:
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code="MISSING_TIME_SIGNATURE",
                message="No active time signature for measure duration validation.",
                node_id=measure.id,
            )
        )
        return

    expected = current_time.measure_duration

    for voice in measure.voices:
        actual = sum((event.duration.fraction for event in voice.events), start=Fraction(0, 1))
        if actual == expected:
            continue

        if actual > expected:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="MEASURE_DURATION_OVERFLOW",
                    message=(
                        f"Voice {voice.id} overfills measure {measure.number}: "
                        f"expected {expected}, got {actual}."
                    ),
                    node_id=measure.id,
                    related_node_ids=tuple(event.id for event in voice.events),
                )
            )
        else:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="MEASURE_DURATION_UNDERFLOW",
                    message=(
                        f"Voice {voice.id} underfills measure {measure.number}: "
                        f"expected {expected}, got {actual}."
                    ),
                    node_id=measure.id,
                    related_node_ids=tuple(event.id for event in voice.events),
                )
            )
