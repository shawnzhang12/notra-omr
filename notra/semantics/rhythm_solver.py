"""
Measure-constrained rhythm solver.

Given a list of symbol candidates with duration hypotheses and a time
signature, finds the best legal assignment using dynamic programming.

Core principle: visual model proposes, measure solver disposes.
False positive is a first-class decoding option.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from notra.semantics import (
    DurationCandidate,
    MeasureDecode,
    SelectedEvent,
    SymbolCandidate,
    VoiceDecode,
    expected_ticks,
)


@dataclass
class _PathState:
    """One state in the DP lattice."""
    score: float
    events: list[tuple[int, DurationCandidate]]  # (candidate_index, chosen_hypothesis)
    skipped: list[int]  # candidate indices marked as false positive


def decode_measure_rhythm(
    candidates: list[SymbolCandidate],
    time_beats: int,
    time_beat_type: int,
    *,
    measure_id: str = "",
    beam_size: int = 256,
) -> MeasureDecode:
    """Decode durations for one measure using DP.

    Args:
        candidates: Symbol candidates sorted left-to-right.
        time_beats: Time signature numerator.
        time_beat_type: Time signature denominator.
        measure_id: Optional measure identifier.
        beam_size: Max paths to keep per DP state.

    Returns:
        MeasureDecode with selected events and diagnostics.
    """
    expected = expected_ticks(time_beats, time_beat_type)

    if not candidates:
        return MeasureDecode(
            measure_id=measure_id,
            expected_ticks=expected,
            voices=[],
            selected_events=[],
            rejected_candidates=[],
            total_score=-float("inf"),
            valid=False,
            diagnostics=["no candidates"],
        )

    # Sort candidates: beamed/flagged first (best visual evidence),
    # then by x-position. Prevents the DP from filling measures with
    # weak candidates and rejecting strong ones that appear later.
    def _cand_priority(c):
        # Flagged candidates get priority (beam/flag evidence is strong)
        flag_bonus = -1000 * c.flag_count if c.flag_count > 0 else 0
        return (flag_bonus, c.x)

    sorted_cands = sorted(candidates, key=_cand_priority)

    # DP: (index, ticks) → best PathState
    dp: dict[tuple[int, int], _PathState] = {
        (0, 0): _PathState(score=0.0, events=[], skipped=[])
    }

    for i, cand in enumerate(sorted_cands):
        new_dp: dict[tuple[int, int], _PathState] = {}

        for (idx, ticks), path in dp.items():
            # Option 1: skip as false positive
            skip_score = path.score + cand.false_positive_score
            key = (i + 1, ticks)
            if key not in new_dp or skip_score > new_dp[key].score:
                new_dp[key] = _PathState(
                    score=skip_score,
                    events=list(path.events),
                    skipped=path.skipped + [i],
                )

            # Option 2: accept one of candidate's duration hypotheses
            hypotheses = cand.duration_candidates
            if not hypotheses:
                hypotheses = [DurationCandidate(
                    480, "quarter", visual_score=-3.0
                )]

            for hyp in hypotheses:
                next_ticks = ticks + hyp.adjusted_ticks
                if next_ticks > expected:
                    continue  # overflow, reject this hypothesis

                # Normalize score by tick count so shorter durations
                # aren't unfairly penalized (fewer candidates = better score).
                tick_frac = hyp.adjusted_ticks / 480.0  # 1.0 = quarter
                hyp_score = path.score + (hyp.visual_score + hyp.grammar_score) * tick_frac
                key = (i + 1, next_ticks)

                if key not in new_dp or hyp_score > new_dp[key].score:
                    new_dp[key] = _PathState(
                        score=hyp_score,
                        events=path.events + [(i, hyp)],
                        skipped=list(path.skipped),
                    )

        # Prune: keep top beam_size paths
        if len(new_dp) > beam_size:
            sorted_states = sorted(new_dp.items(), key=lambda kv: kv[1].score, reverse=True)
            dp = dict(sorted_states[:beam_size])
        else:
            dp = new_dp

    # Find best path that reaches expected ticks.
    # Heavy penalty for underfull measures so the solver works harder
    # to find exact matches, preferring eighths over quarters when needed.
    best_state = None
    best_score = -float("inf")
    underfull_penalty = -100.0

    for (idx, ticks), path in dp.items():
        score = path.score
        if ticks != expected:
            score += underfull_penalty * (expected - ticks) / expected
        if score > best_score or (score == best_score and ticks == expected):
            best_score = score
            best_state = (idx, ticks, path)

    if best_state is None:
        # No valid path found — return best-effort under full
        best_key = max(dp.keys(), key=lambda k: dp[k].score)
        best_path = dp[best_key]
        return MeasureDecode(
            measure_id=measure_id,
            expected_ticks=expected,
            voices=[
                VoiceDecode(
                    voice=1,
                    events=[],
                    total_ticks=best_key[1],
                    is_valid=False,
                )
            ],
            selected_events=[],
            rejected_candidates=[sorted_cands[i].id for i in best_path.skipped],
            total_score=best_path.score,
            valid=False,
            diagnostics=[
                f"no path reaches {expected} ticks; best={best_key[1]} ticks"
            ],
        )

    _, final_ticks, path = best_state
    valid = final_ticks == expected

    # Build selected events
    selected: list[SelectedEvent] = []
    for cand_idx, hyp in path.events:
        cand = sorted_cands[cand_idx]
        selected.append(
            SelectedEvent(
                candidate_id=cand.id,
                duration_ticks=hyp.adjusted_ticks,
                note_type=hyp.note_type,
                dots=hyp.dots,
                voice=1,
                is_rest=cand.is_rest,
                score=hyp.visual_score + hyp.grammar_score,
            )
        )

    diags: list[str] = []
    if not valid:
        diags.append(f"underfull: {final_ticks}/{expected} ticks")

    return MeasureDecode(
        measure_id=measure_id,
        expected_ticks=expected,
        voices=[
            VoiceDecode(
                voice=1,
                events=selected,
                total_ticks=final_ticks,
                is_valid=valid,
            )
        ],
        selected_events=selected,
        rejected_candidates=[sorted_cands[i].id for i in path.skipped],
        total_score=path.score,
        valid=valid,
        diagnostics=diags,
    )


# ---------------------------------------------------------------------------
# Pipeline integration: build candidates from note events
# ---------------------------------------------------------------------------


def build_candidates_from_events(
    note_events: list,
    stem_map: dict[int, Any],
    flag_map: dict[int, int],
    noteheads: list,
    measure_boundaries: list,
    dot_map: dict[int, int] | None = None,
    expected_measure_ticks: int | None = None,
    system_members: list[list[int]] | None = None,
) -> list[list[SymbolCandidate]]:
    """Build per-measure candidate lists from pipeline note events.

    Returns list of candidate lists, one per measure boundary.
    """
    from notra.semantics import (
        SymbolCandidate,
        generate_duration_candidates,
    )

    dot_map = dot_map or {}
    staff_to_system = _staff_to_system(system_members or [])
    boundaries_by_system: dict[int, list] = {}
    for boundary in measure_boundaries:
        system_index = int(getattr(boundary, "system_index", 0))
        boundaries_by_system.setdefault(system_index, []).append(boundary)

    # Group note events by measure boundary
    measures: dict[tuple[int, int], list] = {}
    for ne in note_events:
        event_system = staff_to_system.get(ne.staff_index, 0)
        boundaries = boundaries_by_system.get(event_system, measure_boundaries)
        for mb in boundaries:
            if mb.x_start <= ne.cx < mb.x_end:
                system_index = int(getattr(mb, "system_index", 0))
                measures.setdefault((system_index, mb.measure_number), []).append(ne)
                break

    per_measure_candidates: list[list[SymbolCandidate]] = []
    for system_index, measure_number in sorted(measures.keys()):
        events = measures[(system_index, measure_number)]
        events.sort(key=lambda e: e.cx)

        candidates: list[SymbolCandidate] = []
        for ne in events:
            nh_idx = ne.event_index
            has_stem = nh_idx in stem_map
            flags = flag_map.get(nh_idx, 0)
            dots = dot_map.get(ne.event_index, 0)
            is_filled = False
            if nh_idx < len(noteheads):
                is_filled = noteheads[nh_idx].is_filled

            dur_cands = generate_duration_candidates(
                is_filled=is_filled,
                has_stem=has_stem,
                flag_count=flags,
                dot_count=dots,
                is_rest=ne.is_rest,
            )

            # Adjust false positive score: tiny fragments cheaper to skip
            fp_score = -2.0
            if nh_idx < len(noteheads) and noteheads[nh_idx].area < 50:
                fp_score = -0.8  # small fragment → cheap to skip

            cand = SymbolCandidate(
                id=f"evt_{ne.event_index}",
                bbox=(int(ne.bbox.x0), int(ne.bbox.y0), int(ne.bbox.x1), int(ne.bbox.y1)),
                staff_id=ne.staff_index,
                measure_id=f"s{system_index}_m{measure_number}",
                x=ne.cx,
                y=ne.cy,
                is_filled=is_filled,
                has_stem=has_stem,
                flag_count=flags,
                dot_count=dots,
                is_rest=ne.is_rest,
                duration_candidates=dur_cands,
                false_positive_score=fp_score,
                kind="rest" if ne.is_rest else "note",
            )
            candidates.append(cand)

        if expected_measure_ticks is not None:
            _add_measure_context_duration_candidates(candidates, expected_measure_ticks)
        per_measure_candidates.append(candidates)

    return per_measure_candidates


def _staff_to_system(system_members: list[list[int]]) -> dict[int, int]:
    result: dict[int, int] = {}
    for system_index, members in enumerate(system_members):
        for staff_index in members:
            result[staff_index] = system_index
    return result


def _add_measure_context_duration_candidates(
    candidates: list[SymbolCandidate],
    expected_measure_ticks: int,
) -> None:
    """Add low-priority context durations needed to fill sparse measures."""
    if expected_measure_ticks <= 0 or not candidates:
        return

    note_count = sum(1 for candidate in candidates if not candidate.is_rest)
    if len(candidates) == 1:
        _prepend_duration_if_missing(
            candidates[0],
            expected_measure_ticks,
            visual_score=0.4,
            evidence="single_symbol_measure_fill",
        )

    for candidate in candidates:
        if not candidate.is_rest:
            continue
        full_measure_score = 1.5 if note_count == 0 else -2.0
        _prepend_duration_if_missing(
            candidate,
            expected_measure_ticks,
            visual_score=full_measure_score,
            evidence="full_measure_rest_context",
        )


def _prepend_duration_if_missing(
    candidate: SymbolCandidate,
    duration_ticks: int,
    *,
    visual_score: float,
    evidence: str,
) -> None:
    if any(item.adjusted_ticks == duration_ticks for item in candidate.duration_candidates):
        return
    candidate.duration_candidates.insert(
        0,
        DurationCandidate(
            duration_ticks,
            "whole",
            stem_required=False,
            visual_score=visual_score,
            evidence=evidence,
        ),
    )
