"""Unit tests for pitch spelling representation."""

from __future__ import annotations

import pytest
from notra.ir.note import Pitch


def test_pitch_preserves_spelling_components() -> None:
    pitch = Pitch(step="D", alter=-1, octave=5)
    assert pitch.to_dict() == {"step": "D", "octave": 5, "alter": -1}


def test_pitch_validation() -> None:
    with pytest.raises(ValueError):
        Pitch(step="H", octave=4)

    with pytest.raises(ValueError):
        Pitch(step="C", octave=12)
