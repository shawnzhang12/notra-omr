"""Trainable notehead candidate selection policies.

This module does not try to solve OMR semantics. It selects from a candidate
pool. A threshold policy is ordinary inference. A target-count policy is an
oracle upper bound when the target comes from MusicXML; it must not be reported
as leak-free validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from notra.vision.notehead_pseudolabels import NoteheadPseudoLabel, NoteheadPseudoPage


@dataclass(frozen=True, slots=True)
class ThresholdFitResult:
    """Best global threshold found on a supervised training split."""

    threshold: float
    mean_abs_error: float
    exact_count: int
    fixture_count: int
    total_selected: int
    total_target: int

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "mean_abs_error": self.mean_abs_error,
            "exact_count": self.exact_count,
            "fixture_count": self.fixture_count,
            "total_selected": self.total_selected,
            "total_target": self.total_target,
        }


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Candidate selection result for one page or measure."""

    selected_indices: tuple[int, ...]
    mode: str
    threshold: float
    target_count: int | None
    candidate_count: int
    selected_count: int
    feasible: bool

    @property
    def count_error(self) -> int | None:
        if self.target_count is None:
            return None
        return self.selected_count - self.target_count

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "selected_indices": list(self.selected_indices),
            "mode": self.mode,
            "threshold": self.threshold,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "feasible": self.feasible,
        }
        if self.target_count is not None:
            payload["target_count"] = self.target_count
            payload["count_error"] = self.count_error
        return payload


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate selection metrics over a split."""

    mean_abs_error: float
    exact_count: int
    fixture_count: int
    total_selected: int
    total_target: int
    infeasible_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_abs_error": self.mean_abs_error,
            "exact_count": self.exact_count,
            "fixture_count": self.fixture_count,
            "total_selected": self.total_selected,
            "total_target": self.total_target,
            "infeasible_count": self.infeasible_count,
        }


def threshold_grid(start: float = 0.40, stop: float = 0.98, step: float = 0.01) -> list[float]:
    """Build an inclusive rounded threshold grid."""
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 3))
        current += step
    return values


def fit_global_threshold(
    pages: Iterable[NoteheadPseudoPage],
    *,
    thresholds: Iterable[float] | None = None,
) -> ThresholdFitResult:
    """Fit one confidence threshold by minimizing count MAE on pages with GT counts."""
    page_list = [page for page in pages if page.musicxml_counts is not None]
    if not page_list:
        raise ValueError("at least one page with MusicXML counts is required")

    threshold_values = list(thresholds) if thresholds is not None else threshold_grid()
    if not threshold_values:
        raise ValueError("threshold grid must not be empty")

    best: ThresholdFitResult | None = None
    for threshold in threshold_values:
        result = evaluate_threshold(page_list, threshold=threshold)
        fit = ThresholdFitResult(
            threshold=threshold,
            mean_abs_error=result.mean_abs_error,
            exact_count=result.exact_count,
            fixture_count=result.fixture_count,
            total_selected=result.total_selected,
            total_target=result.total_target,
        )
        if best is None or _fit_sort_key(fit) < _fit_sort_key(best):
            best = fit

    assert best is not None
    return best


def evaluate_threshold(
    pages: Iterable[NoteheadPseudoPage],
    *,
    threshold: float,
) -> EvaluationResult:
    """Evaluate fixed-threshold selection against MusicXML counts."""
    return _evaluate_pages(
        pages,
        lambda page: select_noteheads(
            page.labels,
            threshold=threshold,
            target_count=None,
        ),
    )


def evaluate_oracle_target_count_upper_bound(
    pages: Iterable[NoteheadPseudoPage],
    *,
    threshold: float,
    min_dynamic_confidence: float = 0.0,
) -> EvaluationResult:
    """Evaluate target-count selection using MusicXML counts as an oracle.

    This is a leak by design. It answers only: if a future symbolic solver knew
    the target count, is the visual candidate pool rich enough to select that
    many noteheads?
    """

    def _select(page: NoteheadPseudoPage) -> SelectionResult:
        assert page.musicxml_counts is not None
        return select_noteheads(
            page.labels,
            threshold=threshold,
            target_count=page.musicxml_counts.pitched_noteheads,
            min_dynamic_confidence=min_dynamic_confidence,
        )

    return _evaluate_pages(pages, _select)


def select_noteheads(
    labels: Iterable[NoteheadPseudoLabel],
    *,
    threshold: float,
    target_count: int | None = None,
    min_dynamic_confidence: float = 0.0,
) -> SelectionResult:
    """Select notehead candidates by threshold, with optional dynamic target repair."""
    label_list = list(labels)
    threshold_indices = tuple(label.index for label in label_list if label.confidence >= threshold)

    if target_count is None:
        return SelectionResult(
            selected_indices=threshold_indices,
            mode="threshold",
            threshold=threshold,
            target_count=None,
            candidate_count=len(label_list),
            selected_count=len(threshold_indices),
            feasible=True,
        )

    if len(threshold_indices) == target_count:
        return SelectionResult(
            selected_indices=threshold_indices,
            mode="threshold_exact",
            threshold=threshold,
            target_count=target_count,
            candidate_count=len(label_list),
            selected_count=len(threshold_indices),
            feasible=True,
        )

    ranked = sorted(
        (label for label in label_list if label.confidence >= min_dynamic_confidence),
        key=lambda label: (-label.confidence, label.index),
    )
    if len(ranked) < target_count:
        selected = tuple(label.index for label in ranked)
        return SelectionResult(
            selected_indices=selected,
            mode="infeasible_recall",
            threshold=threshold,
            target_count=target_count,
            candidate_count=len(label_list),
            selected_count=len(selected),
            feasible=False,
        )

    selected = tuple(sorted(label.index for label in ranked[:target_count]))
    return SelectionResult(
        selected_indices=selected,
        mode="dynamic_top_k",
        threshold=threshold,
        target_count=target_count,
        candidate_count=len(label_list),
        selected_count=len(selected),
        feasible=True,
    )


def _fit_sort_key(result: ThresholdFitResult) -> tuple[float, int, float]:
    total_delta = abs(result.total_selected - result.total_target)
    return (result.mean_abs_error, total_delta, -result.threshold)


def _evaluate_pages(
    pages: Iterable[NoteheadPseudoPage],
    selector: Callable[[NoteheadPseudoPage], SelectionResult],
) -> EvaluationResult:
    errors: list[int] = []
    exact_count = 0
    total_selected = 0
    total_target = 0
    infeasible_count = 0
    fixture_count = 0

    for page in pages:
        if page.musicxml_counts is None:
            continue
        result = selector(page)
        target = page.musicxml_counts.pitched_noteheads
        error = result.selected_count - target
        errors.append(abs(error))
        exact_count += int(error == 0)
        total_selected += result.selected_count
        total_target += target
        infeasible_count += int(not result.feasible)
        fixture_count += 1

    if fixture_count == 0:
        raise ValueError("at least one page with MusicXML counts is required")

    return EvaluationResult(
        mean_abs_error=sum(errors) / float(len(errors)),
        exact_count=exact_count,
        fixture_count=fixture_count,
        total_selected=total_selected,
        total_target=total_target,
        infeasible_count=infeasible_count,
    )
