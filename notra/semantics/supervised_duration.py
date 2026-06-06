"""Supervised duration-sequence candidate selection.

This module is intentionally ground-truth-assisted. It is for pseudo-label
generation and upper-bound diagnostics, not leak-free inference.
"""

from __future__ import annotations

from dataclasses import dataclass

from notra.semantics import SymbolCandidate


@dataclass(frozen=True, slots=True)
class SupervisedDurationSelection:
    """Selection of one visual candidate per target duration."""

    target_ticks: tuple[int, ...]
    selected_candidate_ids: tuple[str, ...]
    selected_candidate_indices: tuple[int, ...]
    total_score: float
    valid: bool
    diagnostics: tuple[str, ...] = ()

    @property
    def selected_count(self) -> int:
        return len(self.selected_candidate_ids)


def select_candidates_for_duration_sequence(
    candidates: list[SymbolCandidate],
    target_ticks: tuple[int, ...],
    *,
    expected_ticks: int,
) -> SupervisedDurationSelection:
    """Select a monotonic candidate sequence matching target durations exactly.

    The target durations come from MusicXML or another supervised source. The
    selector uses visual duration evidence as a score, but it does not require
    the current local classifier to have emitted the target duration. This keeps
    the output useful as pseudo-labels for improving that classifier.
    """
    if not target_ticks:
        return SupervisedDurationSelection(
            target_ticks=target_ticks,
            selected_candidate_ids=(),
            selected_candidate_indices=(),
            total_score=0.0,
            valid=True,
        )
    if len(candidates) < len(target_ticks):
        return SupervisedDurationSelection(
            target_ticks=target_ticks,
            selected_candidate_ids=(),
            selected_candidate_indices=(),
            total_score=-float("inf"),
            valid=False,
            diagnostics=(f"candidate shortfall: {len(candidates)} < {len(target_ticks)}",),
        )

    ordered = sorted(enumerate(candidates), key=lambda item: item[1].x)
    target_onsets = _target_onset_fractions(target_ticks, expected_ticks)
    candidate_count = len(ordered)
    target_count = len(target_ticks)

    # dp[(target_index, ordered_candidate_index)] = (score, previous_candidate_index)
    dp: dict[tuple[int, int], tuple[float, int | None]] = {}
    for cand_pos, (_original_idx, candidate) in enumerate(ordered):
        remaining_candidates = candidate_count - cand_pos - 1
        remaining_targets = target_count - 1
        if remaining_candidates < remaining_targets:
            continue
        score = _candidate_score(
            candidate,
            target_ticks[0],
            target_onsets[0],
            measure_x0=ordered[0][1].x,
            measure_x1=ordered[-1][1].x,
        )
        dp[(0, cand_pos)] = (score, None)

    for target_idx in range(1, target_count):
        next_dp: dict[tuple[int, int], tuple[float, int | None]] = {}
        for cand_pos, (_original_idx, candidate) in enumerate(ordered):
            if cand_pos < target_idx:
                continue
            remaining_candidates = candidate_count - cand_pos - 1
            remaining_targets = target_count - target_idx - 1
            if remaining_candidates < remaining_targets:
                continue

            best_score = -float("inf")
            best_prev: int | None = None
            for prev_pos in range(target_idx - 1, cand_pos):
                previous = dp.get((target_idx - 1, prev_pos))
                if previous is None:
                    continue
                transition = _transition_score(
                    ordered[prev_pos][1],
                    candidate,
                    target_ticks[target_idx - 1],
                    expected_ticks,
                )
                score = (
                    previous[0]
                    + transition
                    + _candidate_score(
                        candidate,
                        target_ticks[target_idx],
                        target_onsets[target_idx],
                        measure_x0=ordered[0][1].x,
                        measure_x1=ordered[-1][1].x,
                    )
                )
                if score > best_score:
                    best_score = score
                    best_prev = prev_pos

            if best_prev is not None:
                next_dp[(target_idx, cand_pos)] = (best_score, best_prev)
        dp.update(next_dp)

    final_target_idx = target_count - 1
    final_states = [
        (cand_pos, state)
        for (target_idx, cand_pos), state in dp.items()
        if target_idx == final_target_idx
    ]
    if not final_states:
        return SupervisedDurationSelection(
            target_ticks=target_ticks,
            selected_candidate_ids=(),
            selected_candidate_indices=(),
            total_score=-float("inf"),
            valid=False,
            diagnostics=("no monotonic candidate path",),
        )

    final_pos, (score, _prev) = max(final_states, key=lambda item: item[1][0])
    positions = [final_pos]
    cursor = final_pos
    for target_idx in range(final_target_idx, 0, -1):
        prev = dp[(target_idx, cursor)][1]
        if prev is None:
            break
        positions.append(prev)
        cursor = prev
    positions.reverse()

    original_indices = tuple(ordered[pos][0] for pos in positions)
    candidate_ids = tuple(ordered[pos][1].id for pos in positions)
    return SupervisedDurationSelection(
        target_ticks=target_ticks,
        selected_candidate_ids=candidate_ids,
        selected_candidate_indices=original_indices,
        total_score=score,
        valid=len(candidate_ids) == target_count,
    )


def _target_onset_fractions(
    target_ticks: tuple[int, ...],
    expected_ticks: int,
) -> tuple[float, ...]:
    if expected_ticks <= 0:
        return tuple(0.0 for _item in target_ticks)
    cumulative = 0
    fractions: list[float] = []
    for ticks in target_ticks:
        fractions.append(cumulative / float(expected_ticks))
        cumulative += ticks
    return tuple(fractions)


def _candidate_score(
    candidate: SymbolCandidate,
    target_ticks: int,
    target_onset_fraction: float,
    *,
    measure_x0: float,
    measure_x1: float,
) -> float:
    exact_scores = [
        duration.visual_score + duration.grammar_score
        for duration in candidate.duration_candidates
        if duration.adjusted_ticks == target_ticks
    ]
    duration_score = max(exact_scores) if exact_scores else -0.75
    span = max(1.0, measure_x1 - measure_x0)
    candidate_fraction = (candidate.x - measure_x0) / span
    position_score = -abs(candidate_fraction - target_onset_fraction) * 0.20
    rest_score = 0.15 if candidate.is_rest else 0.0
    return duration_score + position_score + rest_score


def _transition_score(
    previous: SymbolCandidate,
    candidate: SymbolCandidate,
    previous_target_ticks: int,
    expected_ticks: int,
) -> float:
    if candidate.x <= previous.x:
        return -1000.0
    expected_fraction = (
        previous_target_ticks / float(expected_ticks)
        if expected_ticks > 0
        else 0.0
    )
    actual_gap = candidate.x - previous.x
    # Reject only pathological near-duplicates; real eighth notes can be close.
    if actual_gap < 2.0:
        return -4.0
    # The absolute measure width is unknown here, so keep this term weak.
    return min(0.0, expected_fraction) * 0.05
