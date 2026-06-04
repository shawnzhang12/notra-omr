"""Tests for notehead pseudo-label triage contracts."""

from __future__ import annotations

from notra.layout.symbol import NoteheadCandidate
from notra.vision.notehead_policy import select_noteheads
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


def test_dynamic_selection_repairs_threshold_count_when_feasible() -> None:
    config = NoteheadPseudoLabelConfig(
        positive_threshold=0.80,
        uncertain_threshold=0.55,
    )
    labels = tuple(
        NoteheadPseudoLabel.from_candidate(
            idx,
            NoteheadCandidate(
                cx=float(idx),
                cy=20.0,
                bbox=(idx, 17, idx + 2, 23),
                area=30.0,
                is_filled=True,
                staff_step=4.0,
                staff_band_index=0,
                source="connected_component",
                confidence=confidence,
            ),
            config,
        )
        for idx, confidence in enumerate((0.91, 0.84, 0.62, 0.58))
    )

    result = select_noteheads(labels, threshold=0.80, target_count=3)

    assert result.mode == "dynamic_top_k"
    assert result.selected_count == 3
    assert result.count_error == 0
    assert result.feasible


def test_dynamic_selection_reports_infeasible_recall() -> None:
    config = NoteheadPseudoLabelConfig()
    labels = tuple(
        NoteheadPseudoLabel.from_candidate(
            idx,
            NoteheadCandidate(
                cx=float(idx),
                cy=20.0,
                bbox=(idx, 17, idx + 2, 23),
                area=30.0,
                is_filled=True,
                staff_step=4.0,
                staff_band_index=0,
                source="connected_component",
                confidence=0.90,
            ),
            config,
        )
        for idx in range(2)
    )

    result = select_noteheads(labels, threshold=0.80, target_count=3)

    assert result.mode == "infeasible_recall"
    assert result.selected_count == 2
    assert result.count_error == -1
    assert not result.feasible
