"""Tests for notehead pseudo-label triage contracts."""

from __future__ import annotations

from notra.layout.symbol import NoteheadCandidate
from notra.vision.notehead_pseudolabels import (
    NoteheadPseudoLabel,
    NoteheadPseudoLabelConfig,
)


def test_notehead_pseudo_label_thresholds() -> None:
    config = NoteheadPseudoLabelConfig(
        positive_threshold=0.80,
        uncertain_threshold=0.55,
    )
    candidate = NoteheadCandidate(
        cx=10.0,
        cy=20.0,
        bbox=(7, 17, 13, 23),
        area=30.0,
        is_filled=True,
        staff_step=4.0,
        staff_band_index=0,
        source="connected_component",
        confidence=0.62,
    )

    label = NoteheadPseudoLabel.from_candidate(0, candidate, config)

    assert label.label == "uncertain"
    assert label.source == "connected_component"
    assert label.confidence == 0.62
