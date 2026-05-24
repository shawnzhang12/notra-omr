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
