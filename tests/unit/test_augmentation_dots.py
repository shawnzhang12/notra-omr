"""Tests for deterministic augmentation-dot detection."""

from __future__ import annotations

import numpy as np
from notra.layout.dots import detect_augmentation_dots
from notra.layout.staff import StaffBand
from notra.layout.symbol import NoteheadCandidate


def test_detect_augmentation_dot_right_of_line_notehead() -> None:
    ink = _blank_staff()
    ink[34:37, 36:39] = 1

    dot_map, dots = detect_augmentation_dots(ink, [_band()], [_notehead()])

    assert dot_map == {0: 1}
    assert len(dots) == 1
    assert dots[0].event_index == 0


def test_reject_staff_line_fragment_as_augmentation_dot() -> None:
    ink = _blank_staff()
    ink[40, 35:45] = 1

    dot_map, dots = detect_augmentation_dots(ink, [_band()], [_notehead()])

    assert dot_map == {}
    assert dots == []


def test_reject_stem_like_component_as_augmentation_dot() -> None:
    ink = _blank_staff()
    ink[31:49, 37:39] = 1

    dot_map, dots = detect_augmentation_dots(ink, [_band()], [_notehead()])

    assert dot_map == {}
    assert dots == []


def _blank_staff() -> np.ndarray:
    ink = np.zeros((80, 90), dtype=np.uint8)
    for y in _band().line_ys:
        ink[y, 5:80] = 1
    return ink


def _band() -> StaffBand:
    return StaffBand(line_ys=(60, 50, 40, 30, 20), interline_px=10.0)


def _notehead() -> NoteheadCandidate:
    return NoteheadCandidate(
        cx=25.0,
        cy=40.0,
        bbox=(20, 35, 30, 45),
        area=80.0,
        is_filled=True,
        staff_step=4.0,
        staff_band_index=0,
        source="test",
        confidence=1.0,
    )
