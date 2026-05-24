"""Unit tests for advanced note-level feature models."""

from __future__ import annotations

import pytest
from notra.ir.note import Duration, Note, Pitch


def test_note_supports_step5_fields() -> None:
    note = Note(
        id="event-001",
        pitch=Pitch(step="C", octave=4),
        duration=Duration(1, 8),
        ties=("start",),
        slurs=("start",),
        articulations=("staccato",),
        beams=("begin",),
        lyric="la",
        fingering="2",
        chord=False,
        tuplet="start",
        tuplet_ratio=(3, 2),
    )

    payload = note.to_dict()
    assert payload["ties"] == ["start"]
    assert payload["slurs"] == ["start"]
    assert payload["articulations"] == ["staccato"]
    assert payload["beams"] == ["begin"]
    assert payload["lyric"] == "la"
    assert payload["fingering"] == "2"
    assert payload["tuplet"] == "start"


def test_note_rejects_invalid_articulation() -> None:
    with pytest.raises(ValueError):
        Note(
            id="event-001",
            pitch=Pitch(step="C", octave=4),
            duration=Duration(1, 4),
            articulations=("super-legato",),
        )


def test_note_rejects_invalid_tuplet_ratio() -> None:
    with pytest.raises(ValueError):
        Note(
            id="event-001",
            pitch=Pitch(step="C", octave=4),
            duration=Duration(1, 8),
            tuplet="start",
            tuplet_ratio=(0, 2),
        )
