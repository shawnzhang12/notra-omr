"""Unit tests for MusicXML export."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from notra.exporters.musicxml import export_score_to_musicxml
from notra.ir.serialize import score_from_json


def test_musicxml_export_contains_notes_and_signature() -> None:
    fixture_path = Path("tests/fixtures/scores/m001_four_quarters/expected.ir.json")
    score = score_from_json(fixture_path.read_text(encoding="utf-8"))

    xml_text = export_score_to_musicxml(score)
    root = ET.fromstring(xml_text)

    notes = root.findall(".//note")
    assert len(notes) == 4

    beats = root.findtext(".//time/beats")
    beat_type = root.findtext(".//time/beat-type")
    divisions = root.findtext(".//attributes/divisions")

    assert beats == "4"
    assert beat_type == "4"
    assert divisions is not None


def test_musicxml_export_includes_step5_features() -> None:
    fixture_path = Path("tests/fixtures/scores/m010_complex_showcase/expected.ir.json")
    score = score_from_json(fixture_path.read_text(encoding="utf-8"))

    xml_text = export_score_to_musicxml(score)
    root = ET.fromstring(xml_text)

    chord_markers = root.findall(".//note/chord")
    ties = root.findall(".//note/tie")
    slurs = root.findall(".//note/notations/slur")
    articulations = root.findall(".//note/notations/articulations")
    lyrics = root.findall(".//note/lyric/text")
    beams = root.findall(".//note/beam")
    tuplets = root.findall(".//note/notations/tuplet")
    directions = root.findall(".//direction")
    dynamics = root.findall(".//direction/direction-type/dynamics")

    assert chord_markers
    assert ties
    assert slurs
    assert articulations
    assert lyrics
    assert beams
    assert tuplets
    assert directions
    assert dynamics

    note_ids = [note.get("id") for note in root.findall(".//note")]
    assert "event-001" in note_ids
    assert "event-010" in note_ids
