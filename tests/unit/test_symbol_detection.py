"""Tests for symbol-detection geometry contracts."""

from __future__ import annotations

import numpy as np
from notra.layout.staff import StaffBand
from notra.layout.symbol import detect_noteheads


def test_grayscale_notehead_bbox_uses_page_coordinates() -> None:
    gray = np.full((120, 140), 255, dtype=np.uint8)
    gray[58:66, 30:40] = 20
    ink = np.zeros_like(gray, dtype=np.uint8)
    band = StaffBand(line_ys=(40, 50, 60, 70, 80), interline_px=10.0)

    noteheads = detect_noteheads(
        ink,
        [band],
        gray=gray,
        use_grayscale_fallback=True,
    )

    assert len(noteheads) == 1
    assert noteheads[0].cy >= 58
    assert noteheads[0].bbox[1] >= 58
    assert noteheads[0].bbox[3] <= 66
