"""Tests for leak-free notehead measure-selection result scoring."""

from __future__ import annotations

from pathlib import Path

from notra.semantics import DurationCandidate
from notra.vision.notehead_measure_policy import (
    MeasureSelectionResult,
    PageMeasureSelectionResult,
    _add_measure_fill_duration,
    evaluate_leak_free_results,
)


def test_evaluate_leak_free_results_scores_after_inference() -> None:
    page = PageMeasureSelectionResult(
        image_path=Path("tests/fixtures/images/cello/example/page-001.png"),
        threshold=0.89,
        candidate_count=5,
        selected_indices=(0, 1, 2),
        measure_results=(
            MeasureSelectionResult(
                measure_id="s0_m1",
                system_index=0,
                measure_number=1,
                candidate_count=5,
                selected_indices=(0, 1, 2),
                valid=True,
                expected_ticks=1920,
                total_ticks=1920,
                total_score=-1.0,
            ),
        ),
        time_signature=(4, 4),
        staff_count=1,
        system_count=1,
    )

    result = evaluate_leak_free_results(
        [page],
        gt_notehead_counts={"example": 3},
    )

    assert result.exact_page_count == 1
    assert result.mean_abs_notehead_count_error == 0.0
    assert result.valid_measure_count == 1
    assert result.total_measure_count == 1


def test_add_measure_fill_duration_adds_three_four_measure_length() -> None:
    durations = [DurationCandidate(960, "half", visual_score=-0.1)]

    _add_measure_fill_duration(durations, expected_ticks=1440)

    assert durations[0].adjusted_ticks == 1440
    assert durations[0].note_type == "whole"
