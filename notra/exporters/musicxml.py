"""MusicXML exporter for the Step 3/4 deterministic IR slice."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from xml.etree import ElementTree as ET

from notra.ir.measure import MeasureAttributes, Voice
from notra.ir.note import Duration, Note
from notra.ir.rest import Rest
from notra.ir.score import Part, Score


def export_score_to_musicxml(score: Score) -> str:
    """Export a Score to a MusicXML score-partwise document string."""
    divisions = _score_divisions(score)

    root = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = score.title

    part_list = ET.SubElement(root, "part-list")
    for part in score.parts:
        score_part = ET.SubElement(part_list, "score-part", id=part.id)
        ET.SubElement(score_part, "part-name").text = part.name

    for part in score.parts:
        part_elem = ET.SubElement(root, "part", id=part.id)
        _append_part_measures(part_elem, part, divisions)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _append_part_measures(part_elem: ET.Element, part: Part, default_divisions: int) -> None:
    for index, measure in enumerate(part.measures):
        measure_elem = ET.SubElement(part_elem, "measure", number=str(measure.number))

        if index == 0 or measure.attributes is not None:
            _append_attributes(measure_elem, measure.attributes, default_divisions)

        for voice_index, voice in enumerate(measure.voices, start=1):
            _append_voice_events(measure_elem, voice, voice_index, default_divisions)

        if measure.barline is not None and measure.barline.style != "regular":
            ET.SubElement(measure_elem, "barline", location="right")


def _append_attributes(
    measure_elem: ET.Element,
    attributes: MeasureAttributes | None,
    default_divisions: int,
) -> None:
    attr_elem = ET.SubElement(measure_elem, "attributes")
    divisions = attributes.divisions if attributes and attributes.divisions else default_divisions
    ET.SubElement(attr_elem, "divisions").text = str(divisions)

    if attributes and attributes.key is not None:
        key_elem = ET.SubElement(attr_elem, "key")
        ET.SubElement(key_elem, "fifths").text = str(attributes.key.fifths)
        ET.SubElement(key_elem, "mode").text = attributes.key.mode

    if attributes and attributes.time is not None:
        time_elem = ET.SubElement(attr_elem, "time")
        ET.SubElement(time_elem, "beats").text = str(attributes.time.beats)
        ET.SubElement(time_elem, "beat-type").text = str(attributes.time.beat_type)

    if attributes and attributes.clef is not None:
        clef_elem = ET.SubElement(attr_elem, "clef")
        ET.SubElement(clef_elem, "sign").text = attributes.clef.sign
        ET.SubElement(clef_elem, "line").text = str(attributes.clef.line)


def _append_voice_events(
    measure_elem: ET.Element,
    voice: Voice,
    fallback_voice: int,
    divisions: int,
) -> None:
    voice_number = _voice_number(voice.id, fallback_voice)

    for event in voice.events:
        note_elem = ET.SubElement(measure_elem, "note")
        if isinstance(event, Rest):
            ET.SubElement(note_elem, "rest")
        elif isinstance(event, Note):
            pitch_elem = ET.SubElement(note_elem, "pitch")
            ET.SubElement(pitch_elem, "step").text = event.pitch.step
            if event.pitch.alter != 0:
                ET.SubElement(pitch_elem, "alter").text = str(event.pitch.alter)
            ET.SubElement(pitch_elem, "octave").text = str(event.pitch.octave)
            _append_ties(note_elem, event)

        ET.SubElement(note_elem, "duration").text = str(_duration_units(event.duration, divisions))
        ET.SubElement(note_elem, "voice").text = str(event.voice or voice_number)

        note_type, dot_count = _duration_type_and_dots(event.duration.fraction)
        ET.SubElement(note_elem, "type").text = note_type
        for _ in range(dot_count):
            ET.SubElement(note_elem, "dot")


def _append_ties(note_elem: ET.Element, note: Note) -> None:
    if not note.ties:
        return

    notation_elem: ET.Element | None = None
    for tie in note.ties:
        if tie in {"start", "stop"}:
            ET.SubElement(note_elem, "tie", type=tie)
            if notation_elem is None:
                notation_elem = ET.SubElement(note_elem, "notations")
            ET.SubElement(notation_elem, "tied", type=tie)


def _duration_units(duration: Duration, divisions: int) -> int:
    units = duration.fraction * 4 * divisions
    if units.denominator != 1:
        raise ValueError(
            "duration cannot be represented exactly with chosen divisions: "
            f"{duration.numerator}/{duration.denominator}"
        )
    return units.numerator


def _duration_type_and_dots(fraction: Fraction) -> tuple[str, int]:
    type_map = {
        Fraction(1, 1): "whole",
        Fraction(1, 2): "half",
        Fraction(1, 4): "quarter",
        Fraction(1, 8): "eighth",
        Fraction(1, 16): "16th",
        Fraction(1, 32): "32nd",
    }
    if fraction in type_map:
        return type_map[fraction], 0

    base = fraction * Fraction(2, 3)
    if base in type_map:
        return type_map[base], 1

    return "quarter", 0


def _score_divisions(score: Score) -> int:
    denominators: set[int] = set()
    for part in score.parts:
        for measure in part.measures:
            for voice in measure.voices:
                for event in voice.events:
                    denominators.add(event.duration.denominator)

    if not denominators:
        return 1

    result = 1
    for denominator in denominators:
        result = _lcm(result, denominator)
    return result


def _lcm(left: int, right: int) -> int:
    return left * right // gcd(left, right)


def _voice_number(voice_id: str, fallback: int) -> int:
    digits = "".join(char for char in voice_id if char.isdigit())
    if digits:
        return int(digits)
    return fallback
