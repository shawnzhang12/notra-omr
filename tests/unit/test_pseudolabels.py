"""Tests for deterministic segmentation pseudo-label generation."""

from __future__ import annotations

import numpy as np
from notra.layout.beam_detector import BeamCandidate
from notra.layout.staff import StaffBand
from notra.layout.stem_detector import StemCandidate
from notra.layout.symbol import NoteheadCandidate
from notra.vision.pseudolabels import (
    PseudoMaskConfig,
    build_pseudo_segmentation_mask,
    class_pixel_counts,
    colorize_mask,
)
from notra.vision.schema import SegmentationClass


def test_build_pseudo_segmentation_mask_layers_core_symbols() -> None:
    ink = np.zeros((80, 120), dtype=np.uint8)
    ink[18:43, 89:92] = 1
    band = StaffBand(line_ys=(20, 25, 30, 35, 40), interline_px=5.0)
    notehead = NoteheadCandidate(
        cx=50.0,
        cy=30.0,
        bbox=(46, 27, 54, 33),
        area=40.0,
        is_filled=True,
        staff_step=4.0,
        staff_band_index=0,
    )
    stem = StemCandidate(
        x0=55,
        y0=12,
        x1=56,
        y1=31,
        x_center=55.5,
        height=20,
        width=2,
        staff_id=0,
        direction="up",
    )
    beam = BeamCandidate(
        x0=55,
        y0=12,
        x1=80,
        y1=14,
        thickness=0.4,
        level=1,
        staff_id=0,
        connected_stems=[0],
        score=1.0,
    )

    mask = build_pseudo_segmentation_mask(
        ink=ink,
        staff_bands=[band],
        noteheads=[notehead],
        stems=[stem],
        beams=[beam],
        barline_xs=[90.0],
        system_members=[[0]],
        config=PseudoMaskConfig(include_clef_region=False),
    )

    assert mask[20, 10] == int(SegmentationClass.STAFF_LINE)
    assert mask[30, 50] == int(SegmentationClass.NOTEHEAD_FILLED)
    assert mask[20, 55] == int(SegmentationClass.STEM)
    assert mask[13, 70] == int(SegmentationClass.BEAM)
    assert mask[30, 90] == int(SegmentationClass.BARLINE)

    counts = class_pixel_counts(mask)
    assert counts["staff_line"] > 0
    assert counts["notehead_filled"] > 0


def test_colorize_mask_returns_rgb_image() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = int(SegmentationClass.NOTEHEAD_FILLED)

    image = colorize_mask(mask)

    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_build_pseudo_segmentation_mask_filters_low_confidence_noteheads() -> None:
    ink = np.zeros((80, 120), dtype=np.uint8)
    band = StaffBand(line_ys=(20, 25, 30, 35, 40), interline_px=5.0)
    low_confidence_notehead = NoteheadCandidate(
        cx=50.0,
        cy=30.0,
        bbox=(46, 27, 54, 33),
        area=40.0,
        is_filled=True,
        staff_step=4.0,
        staff_band_index=0,
        source="connected_component",
        confidence=0.20,
    )

    mask = build_pseudo_segmentation_mask(
        ink=ink,
        staff_bands=[band],
        noteheads=[low_confidence_notehead],
        config=PseudoMaskConfig(include_clef_region=False, min_notehead_confidence=0.80),
    )

    assert int(SegmentationClass.NOTEHEAD_FILLED) not in set(mask.ravel().tolist())
