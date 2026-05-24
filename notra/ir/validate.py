"""Semantic validation for Notra score IR."""

from __future__ import annotations

from fractions import Fraction

from notra.core.errors import Severity, ValidationIssue, ValidationReport
from notra.ir.measure import Measure
from notra.ir.note import Note
from notra.ir.score import Part, Score
from notra.ir.time import TimeSignature


def validate_score(score: Score) -> ValidationReport:
    """Validate score-level semantic constraints and return a report."""
    report = ValidationReport()
    seen_event_ids: set[str] = set()
    for part in score.parts:
        _validate_part(part, report, seen_event_ids)
    return report


def _validate_part(part: Part, report: ValidationReport, seen_event_ids: set[str]) -> None:
    current_time: TimeSignature | None = None
    seen_measure_numbers: set[int] = set()
    last_measure_number: int | None = None
    active_ties: set[tuple[str, str]] = set()

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

        if last_measure_number is not None and measure.number <= last_measure_number:
            report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="NON_MONOTONIC_MEASURE_ORDER",
                    message=(
                        f"Measure numbers are not strictly increasing in part {part.id}: "
                        f"{last_measure_number} -> {measure.number}."
                    ),
                    node_id=measure.id,
                )
            )
        last_measure_number = measure.number

        if measure.attributes and measure.attributes.time is not None:
            current_time = measure.attributes.time

        _validate_measure(measure, report, current_time, seen_event_ids, active_ties)

    for voice_id, pitch_key in sorted(active_ties):
        report.add(
            ValidationIssue(
                severity=Severity.WARNING,
                code="UNCLOSED_TIE_AT_PART_END",
                message=(
                    "Tie start does not have a matching stop by part end: "
                    f"voice={voice_id}, pitch={pitch_key}."
                ),
                node_id=part.id,
            )
        )


def _validate_measure(
    measure: Measure,
    report: ValidationReport,
    current_time: TimeSignature | None,
    seen_event_ids: set[str],
    active_ties: set[tuple[str, str]],
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

    seen_voice_ids: set[str] = set()
    for voice in measure.voices:
        if voice.id in seen_voice_ids:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="DUPLICATE_VOICE_ID",
                    message=f"Duplicate voice id {voice.id!r} in measure {measure.number}.",
                    node_id=measure.id,
                )
            )
        seen_voice_ids.add(voice.id)

        if not voice.events:
            report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="EMPTY_VOICE",
                    message=f"Voice {voice.id!r} in measure {measure.number} has no events.",
                    node_id=measure.id,
                )
            )

        expected_voice = _voice_id_number(voice.id)
        chord_anchor_duration: Fraction | None = None
        active_tuplet_ratio: tuple[int, int] | None = None
        active_slurs = 0
        actual = sum(
            (Fraction(0, 1) if isinstance(event, Note) and event.chord else event.duration.fraction)
            for event in voice.events
        )

        for event in voice.events:
            if event.id in seen_event_ids:
                report.add(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="DUPLICATE_EVENT_ID",
                        message=f"Duplicate event id {event.id!r} detected in score.",
                        node_id=measure.id,
                        related_node_ids=(event.id,),
                    )
                )
            else:
                seen_event_ids.add(event.id)

            if expected_voice is not None and event.voice != expected_voice:
                report.add(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        code="EVENT_VOICE_MISMATCH",
                        message=(
                            f"Event {event.id!r} has voice={event.voice}, but container "
                            f"voice id {voice.id!r} implies {expected_voice}."
                        ),
                        node_id=measure.id,
                        related_node_ids=(event.id,),
                    )
                )

            if isinstance(event, Note):
                _validate_note_chord_group(
                    event, voice.id, measure.id, report, chord_anchor_duration
                )
                if not event.chord:
                    chord_anchor_duration = event.duration.fraction
                _validate_note_tuplet(event, voice.id, measure.id, report, active_tuplet_ratio)
                active_tuplet_ratio = _next_active_tuplet_ratio(event, active_tuplet_ratio)
                active_slurs = _validate_note_slurs(
                    event,
                    voice.id,
                    measure.id,
                    report,
                    active_slurs,
                )
                _validate_note_ties(event, voice.id, measure.id, report, active_ties)

        if active_tuplet_ratio is not None:
            report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="UNCLOSED_TUPLET_AT_MEASURE_END",
                    message=(
                        f"Voice {voice.id!r} ends measure {measure.number} with an open tuplet "
                        f"ratio {active_tuplet_ratio[0]}:{active_tuplet_ratio[1]}."
                    ),
                    node_id=measure.id,
                )
            )
        if active_slurs > 0:
            report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="UNCLOSED_SLUR_AT_MEASURE_END",
                    message=(
                        f"Voice {voice.id!r} ends measure {measure.number} with {active_slurs} "
                        "unclosed slur marker(s)."
                    ),
                    node_id=measure.id,
                )
            )

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


def _validate_note_ties(
    note: Note,
    voice_id: str,
    measure_id: str,
    report: ValidationReport,
    active_ties: set[tuple[str, str]],
) -> None:
    tie_types = set(note.ties)
    if not tie_types:
        return

    pitch_key = _note_pitch_key(note)
    tie_key = (voice_id, pitch_key)

    if "continue" in tie_types and ("start" in tie_types or "stop" in tie_types):
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code="INVALID_TIE_ENCODING",
                message=(
                    f"Note {note.id!r} mixes 'continue' with 'start'/'stop'. "
                    "Use either continue or explicit start+stop."
                ),
                node_id=measure_id,
                related_node_ids=(note.id,),
            )
        )
        return

    if "continue" in tie_types:
        if tie_key not in active_ties:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="TIE_CONTINUE_WITHOUT_START",
                    message=f"Note {note.id!r} continues a tie that was never opened.",
                    node_id=measure_id,
                    related_node_ids=(note.id,),
                )
            )
        return

    has_start = "start" in tie_types
    has_stop = "stop" in tie_types

    if has_stop and tie_key not in active_ties:
        code = "TIE_STOP_WITHOUT_START"
        message = f"Note {note.id!r} stops a tie that was never opened for pitch {pitch_key}."
        if any(active_voice == voice_id for active_voice, _ in active_ties):
            code = "TIE_STOP_PITCH_MISMATCH"
            message = (
                f"Note {note.id!r} stops tie for pitch {pitch_key}, "
                "but no matching pitch tie is open "
                f"in voice {voice_id!r}."
            )
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code=code,
                message=message,
                node_id=measure_id,
                related_node_ids=(note.id,),
            )
        )

    if has_stop and tie_key in active_ties:
        active_ties.remove(tie_key)

    if has_start:
        if tie_key in active_ties:
            report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="TIE_START_WHILE_OPEN",
                    message=f"Note {note.id!r} starts a tie that is already open.",
                    node_id=measure_id,
                    related_node_ids=(note.id,),
                )
            )
        active_ties.add(tie_key)


def _validate_note_chord_group(
    note: Note,
    voice_id: str,
    measure_id: str,
    report: ValidationReport,
    chord_anchor_duration: Fraction | None,
) -> None:
    if not note.chord:
        return
    if chord_anchor_duration is None:
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code="CHORD_WITHOUT_ANCHOR",
                message=(
                    f"Note {note.id!r} is marked as chord tone without a preceding anchor "
                    f"note in voice {voice_id!r}."
                ),
                node_id=measure_id,
                related_node_ids=(note.id,),
            )
        )
        return
    if note.duration.fraction > chord_anchor_duration:
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code="CHORD_DURATION_EXCEEDS_ANCHOR",
                message=(
                    f"Chord tone {note.id!r} duration {note.duration.fraction} exceeds anchor "
                    f"duration {chord_anchor_duration} in voice {voice_id!r}."
                ),
                node_id=measure_id,
                related_node_ids=(note.id,),
            )
        )


def _validate_note_tuplet(
    note: Note,
    voice_id: str,
    measure_id: str,
    report: ValidationReport,
    active_tuplet_ratio: tuple[int, int] | None,
) -> None:
    if note.tuplet is None:
        if active_tuplet_ratio is not None and note.tuplet_ratio is None:
            report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="MISSING_TUPLET_RATIO_IN_ACTIVE_GROUP",
                    message=(
                        f"Note {note.id!r} in voice {voice_id!r} is inside an active tuplet "
                        "group but lacks tuplet_ratio."
                    ),
                    node_id=measure_id,
                    related_node_ids=(note.id,),
                )
            )
        return

    if note.tuplet == "start":
        if active_tuplet_ratio is not None:
            report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="TUPLET_START_WHILE_OPEN",
                    message=f"Note {note.id!r} starts a new tuplet while one is already open.",
                    node_id=measure_id,
                    related_node_ids=(note.id,),
                )
            )
        if note.tuplet_ratio is None:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="TUPLET_START_MISSING_RATIO",
                    message=f"Tuplet start on note {note.id!r} must define tuplet_ratio.",
                    node_id=measure_id,
                    related_node_ids=(note.id,),
                )
            )
        return

    if note.tuplet == "stop":
        if active_tuplet_ratio is None:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="TUPLET_STOP_WITHOUT_START",
                    message=f"Tuplet stop on note {note.id!r} has no active tuplet to close.",
                    node_id=measure_id,
                    related_node_ids=(note.id,),
                )
            )
            return
        if note.tuplet_ratio is not None and note.tuplet_ratio != active_tuplet_ratio:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="TUPLET_RATIO_MISMATCH",
                    message=(
                        f"Tuplet stop on note {note.id!r} has ratio {note.tuplet_ratio[0]}:"
                        f"{note.tuplet_ratio[1]}, expected {active_tuplet_ratio[0]}:"
                        f"{active_tuplet_ratio[1]}."
                    ),
                    node_id=measure_id,
                    related_node_ids=(note.id,),
                )
            )


def _next_active_tuplet_ratio(
    note: Note,
    active_tuplet_ratio: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if note.tuplet == "start":
        return note.tuplet_ratio
    if note.tuplet == "stop":
        return None
    return active_tuplet_ratio


def _validate_note_slurs(
    note: Note,
    voice_id: str,
    measure_id: str,
    report: ValidationReport,
    active_slurs: int,
) -> int:
    updated = active_slurs
    for marker in note.slurs:
        if marker == "start":
            updated += 1
            continue
        if marker == "continue":
            if updated <= 0:
                report.add(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="SLUR_CONTINUE_WITHOUT_START",
                        message=f"Slur continue on note {note.id!r} without active slur.",
                        node_id=measure_id,
                        related_node_ids=(note.id,),
                    )
                )
            continue
        if marker == "stop":
            if updated <= 0:
                report.add(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="SLUR_STOP_WITHOUT_START",
                        message=f"Slur stop on note {note.id!r} without active slur.",
                        node_id=measure_id,
                        related_node_ids=(note.id,),
                    )
                )
                continue
            updated -= 1
    return updated


def _note_pitch_key(note: Note) -> str:
    pitch = note.pitch
    return f"{pitch.step}{pitch.alter:+d}:{pitch.octave}"


def _voice_id_number(voice_id: str) -> int | None:
    digits = "".join(char for char in voice_id if char.isdigit())
    if not digits:
        return None
    return int(digits)
