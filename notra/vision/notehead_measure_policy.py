"""Leak-free notehead selection from measure-duration constraints.

This policy uses no MusicXML counts during inference. It selects candidates by
running a rhythm decoder per detected measure. MusicXML may be used later by
evaluation scripts to score the selected result, but not to choose candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from notra.layout.annotations import MeasureBoundary
from notra.layout.measure import estimate_staff_x_extent
from notra.layout.stem_detector import detect_stems_global
from notra.layout.symbol import NoteheadCandidate
from notra.pipeline import stages
from notra.pipeline.config import PipelineConfig
from notra.semantics import DurationCandidate, SymbolCandidate, generate_duration_candidates
from notra.semantics import expected_ticks as expected_measure_ticks
from notra.semantics.rhythm_solver import decode_measure_rhythm
from notra.vision.notehead_pseudolabels import (
    NoteheadPseudoLabel,
    NoteheadPseudoLabelConfig,
)


@dataclass(frozen=True, slots=True)
class MeasureSelectionConfig:
    """Controls leak-free measure-level candidate selection."""

    threshold: float
    include_relaxed_rescue: bool = True
    min_candidate_confidence: float = 0.0
    confidence_score_scale: float = 6.0
    high_confidence_skip_penalty: float = 5.0
    low_confidence_skip_penalty: float = 0.25


@dataclass(frozen=True, slots=True)
class MeasureSelectionResult:
    """One measure decoded without ground-truth count targets."""

    measure_id: str
    system_index: int
    measure_number: int
    candidate_count: int
    selected_indices: tuple[int, ...]
    valid: bool
    expected_ticks: int
    total_ticks: int
    total_score: float
    diagnostics: tuple[str, ...] = ()

    @property
    def selected_count(self) -> int:
        return len(self.selected_indices)

    def to_dict(self) -> dict[str, object]:
        return {
            "measure_id": self.measure_id,
            "system_index": self.system_index,
            "measure_number": self.measure_number,
            "candidate_count": self.candidate_count,
            "selected_indices": list(self.selected_indices),
            "selected_count": self.selected_count,
            "valid": self.valid,
            "expected_ticks": self.expected_ticks,
            "total_ticks": self.total_ticks,
            "total_score": self.total_score,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class PageMeasureSelectionResult:
    """Leak-free notehead selection result for one page."""

    image_path: Path
    threshold: float
    candidate_count: int
    selected_indices: tuple[int, ...]
    measure_results: tuple[MeasureSelectionResult, ...]
    time_signature: tuple[int, int]
    staff_count: int
    system_count: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def selected_count(self) -> int:
        return len(self.selected_indices)

    @property
    def measure_count(self) -> int:
        return len(self.measure_results)

    @property
    def valid_measure_count(self) -> int:
        return sum(1 for item in self.measure_results if item.valid)

    @property
    def all_measures_valid(self) -> bool:
        return self.measure_count > 0 and self.valid_measure_count == self.measure_count

    def to_summary(self) -> dict[str, object]:
        return {
            "image_path": str(self.image_path),
            "threshold": self.threshold,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "measure_count": self.measure_count,
            "valid_measure_count": self.valid_measure_count,
            "all_measures_valid": self.all_measures_valid,
            "time_signature": f"{self.time_signature[0]}/{self.time_signature[1]}",
            "staff_count": self.staff_count,
            "system_count": self.system_count,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class LeakFreeEvaluationResult:
    """Evaluation metrics where ground truth is used only after inference."""

    fixture_count: int
    exact_page_count: int
    mean_abs_notehead_count_error: float
    total_selected_noteheads: int
    total_gt_noteheads: int
    valid_measure_count: int
    total_measure_count: int
    all_measures_valid_count: int

    @property
    def valid_measure_rate(self) -> float:
        if self.total_measure_count == 0:
            return 0.0
        return self.valid_measure_count / float(self.total_measure_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture_count": self.fixture_count,
            "exact_page_count": self.exact_page_count,
            "mean_abs_notehead_count_error": self.mean_abs_notehead_count_error,
            "total_selected_noteheads": self.total_selected_noteheads,
            "total_gt_noteheads": self.total_gt_noteheads,
            "valid_measure_count": self.valid_measure_count,
            "total_measure_count": self.total_measure_count,
            "valid_measure_rate": self.valid_measure_rate,
            "all_measures_valid_count": self.all_measures_valid_count,
        }


def solve_noteheads_by_measure(
    image_path: str | Path,
    *,
    config: MeasureSelectionConfig,
    pipeline_config: PipelineConfig | None = None,
) -> PageMeasureSelectionResult:
    """Select notehead candidates by measure-duration validity without GT counts."""
    image_path = Path(image_path)
    pipeline_config = pipeline_config or PipelineConfig.for_image(image_path)
    ctx: dict[str, Any] = {
        "image_path": str(image_path),
        "errors": [],
        "warnings": [],
        "metrics": {},
    }
    ctx.update(pipeline_config.to_context())

    for stage_fn in (
        stages.load_image_stage,
        stages.detect_layout_stage,
        stages.detect_clefs_stage,
        stages.detect_noteheads_stage,
        stages.detect_time_signature_stage,
        stages.detect_rests_stage,
    ):
        try:
            stage_fn(ctx)
        except Exception as exc:  # pragma: no cover - artifact path
            ctx.setdefault("errors", []).append(f"{stage_fn.__name__}: {exc}")

    noteheads = tuple(ctx.get("notehead_candidates", []))
    rest_candidates = tuple(ctx.get("rest_candidates", []))
    if config.include_relaxed_rescue:
        noteheads = _with_relaxed_rescue_candidates(noteheads, ctx)

    labels = tuple(
        NoteheadPseudoLabel.from_candidate(
            idx,
            candidate,
            NoteheadPseudoLabelConfig(
                positive_threshold=config.threshold,
                uncertain_threshold=max(0.0, config.threshold - 0.20),
            ),
        )
        for idx, candidate in enumerate(noteheads)
    )
    candidate_indices = {
        label.index for label in labels if label.confidence >= config.min_candidate_confidence
    }

    measure_boundaries = _build_measure_boundaries(ctx)
    stem_map, flag_map = _detect_stems_and_flags(ctx, list(noteheads))
    time_beats = int(ctx.get("_structural_time_beats", 4) or 4)
    time_beat_type = int(ctx.get("_structural_time_beat_type", 4) or 4)

    cands_by_measure = _build_symbol_candidates_by_measure(
        labels=labels,
        noteheads=noteheads,
        allowed_indices=candidate_indices,
        measure_boundaries=measure_boundaries,
        system_members=ctx.get("system_members", []),
        stem_map=stem_map,
        flag_map=flag_map,
        config=config,
        rest_candidates=rest_candidates,
        expected_ticks=expected_measure_ticks(time_beats, time_beat_type),
    )

    measure_results: list[MeasureSelectionResult] = []
    selected: set[int] = set()
    for boundary in measure_boundaries:
        measure_id = _measure_id(boundary)
        candidates = cands_by_measure.get(measure_id, [])
        decode = decode_measure_rhythm(
            candidates,
            time_beats=time_beats,
            time_beat_type=time_beat_type,
            measure_id=measure_id,
        )
        selected_indices = tuple(
            sorted(
                int(event.candidate_id.split("_", maxsplit=1)[1])
                for event in decode.selected_events
                if event.candidate_id.startswith("nh_")
            )
        )
        selected.update(selected_indices)
        total_ticks = decode.voices[0].total_ticks if decode.voices else 0
        measure_results.append(
            MeasureSelectionResult(
                measure_id=measure_id,
                system_index=boundary.system_index,
                measure_number=boundary.measure_number,
                candidate_count=len(candidates),
                selected_indices=selected_indices,
                valid=decode.valid,
                expected_ticks=decode.expected_ticks,
                total_ticks=total_ticks,
                total_score=decode.total_score,
                diagnostics=tuple(decode.diagnostics),
            )
        )

    return PageMeasureSelectionResult(
        image_path=image_path,
        threshold=config.threshold,
        candidate_count=len(labels),
        selected_indices=tuple(sorted(selected)),
        measure_results=tuple(measure_results),
        time_signature=(time_beats, time_beat_type),
        staff_count=len(ctx.get("staff_bands", [])),
        system_count=len(ctx.get("system_members", [])),
        errors=tuple(str(item) for item in ctx.get("errors", [])),
        warnings=tuple(str(item) for item in ctx.get("warnings", [])),
    )


def evaluate_leak_free_results(
    pages: list[PageMeasureSelectionResult],
    *,
    gt_notehead_counts: dict[str, int],
) -> LeakFreeEvaluationResult:
    """Score leak-free results after inference using fixture MusicXML counts."""
    errors: list[int] = []
    exact_pages = 0
    total_selected = 0
    total_gt = 0
    valid_measure_count = 0
    total_measure_count = 0
    all_valid_pages = 0

    for page in pages:
        name = page.image_path.parent.name
        if name not in gt_notehead_counts:
            continue
        gt = gt_notehead_counts[name]
        selected = page.selected_count
        error = selected - gt
        errors.append(abs(error))
        exact_pages += int(error == 0)
        total_selected += selected
        total_gt += gt
        valid_measure_count += page.valid_measure_count
        total_measure_count += page.measure_count
        all_valid_pages += int(page.all_measures_valid)

    if not errors:
        raise ValueError("at least one page with a GT count is required")

    return LeakFreeEvaluationResult(
        fixture_count=len(errors),
        exact_page_count=exact_pages,
        mean_abs_notehead_count_error=sum(errors) / float(len(errors)),
        total_selected_noteheads=total_selected,
        total_gt_noteheads=total_gt,
        valid_measure_count=valid_measure_count,
        total_measure_count=total_measure_count,
        all_measures_valid_count=all_valid_pages,
    )


def _with_relaxed_rescue_candidates(
    candidates: tuple[NoteheadCandidate, ...],
    ctx: dict[str, Any],
) -> tuple[NoteheadCandidate, ...]:
    ink = ctx.get("ink")
    gray = ctx.get("gray")
    bands = ctx.get("staff_bands", [])
    if ink is None or gray is None or not bands:
        return candidates

    from notra.layout.symbol import detect_noteheads
    from notra.vision.notehead_pseudolabels import _merge_notehead_candidates

    relaxed = tuple(
        detect_noteheads(
            ink,
            bands,
            gray=gray,
            use_grayscale_fallback=True,
            use_line_position_pass=True,
        )
    )
    interline = float(ctx.get("interline_px", 0.0) or 0.0)
    return _merge_notehead_candidates(candidates, relaxed, interline=interline)


def _build_measure_boundaries(ctx: dict[str, Any]) -> list[MeasureBoundary]:
    gray: np.ndarray | None = ctx.get("gray")
    bands = ctx.get("staff_bands", [])
    system_members: list[list[int]] = ctx.get("system_members", [])
    barline_by_system: dict[int, list[float]] = ctx.get("barline_by_system", {})
    image_width = float(ctx.get("image_width", 0) or 0)

    if gray is None or not bands:
        return []

    systems = system_members or [list(range(len(bands)))]
    boundaries: list[MeasureBoundary] = []
    for sys_idx, members in enumerate(systems):
        spans = [
            estimate_staff_x_extent(gray, bands[staff_idx])
            for staff_idx in members
            if 0 <= staff_idx < len(bands)
        ]
        if spans:
            left = float(np.median([span[0] for span in spans]))
            right = float(np.median([span[1] for span in spans]))
        else:
            left = 0.0
            right = image_width

        bars = sorted(barline_by_system.get(sys_idx, []))
        xs = [left] + bars
        if not bars or right - bars[-1] > max(10.0, float(ctx.get("interline_px", 12.0)) * 2.0):
            xs.append(right)
        for idx, (x0, x1) in enumerate(zip(xs, xs[1:], strict=False), start=1):
            if x1 - x0 < 10.0:
                continue
            boundaries.append(
                MeasureBoundary(
                    measure_number=idx,
                    x_start=x0,
                    x_end=x1,
                    staff_index=0,
                    system_index=sys_idx,
                )
            )
    return boundaries


def _detect_stems_and_flags(
    ctx: dict[str, Any],
    noteheads: list[NoteheadCandidate],
) -> tuple[dict[int, Any], dict[int, int]]:
    ink = ctx.get("ink")
    bands = ctx.get("staff_bands", [])
    if ink is None or not noteheads or not bands:
        return {}, {}

    from notra.layout.beam_detector import detect_beams
    from notra.layout.symbol import StemCandidate

    global_stems, stem_map_raw = detect_stems_global(ink, bands, noteheads)
    pipeline_stem_map: dict[int, Any] = {}
    for nh_idx, gs in stem_map_raw.items():
        pipeline_stem_map[nh_idx] = StemCandidate(
            notehead_cx=noteheads[nh_idx].cx,
            notehead_cy=noteheads[nh_idx].cy,
            direction=gs.direction,
            top_y=gs.y0 if gs.direction == "up" else gs.y1,
            bottom_y=gs.y1 if gs.direction == "up" else gs.y0,
            center_x=gs.x_center,
            length_px=gs.height,
        )

    interline = float(ctx.get("interline_px", 0.0) or np.median([b.interline_px for b in bands]))
    beams = detect_beams(ink, bands, list(stem_map_raw.values()), interline)
    stem_list = list(stem_map_raw.values())
    stem_idx_to_nh: dict[int, int] = {}
    for si, stem in enumerate(stem_list):
        for nh_idx, gs in stem_map_raw.items():
            if gs is stem:
                stem_idx_to_nh[si] = nh_idx
                break

    flag_map: dict[int, int] = {}
    for beam in beams:
        for si in beam.connected_stems:
            connected_notehead_idx = stem_idx_to_nh.get(si)
            if connected_notehead_idx is not None:
                flag_map[connected_notehead_idx] = max(
                    flag_map.get(connected_notehead_idx, 0), beam.level
                )
    _ = global_stems
    return pipeline_stem_map, flag_map


def _build_symbol_candidates_by_measure(
    *,
    labels: tuple[NoteheadPseudoLabel, ...],
    noteheads: tuple[NoteheadCandidate, ...],
    allowed_indices: set[int],
    measure_boundaries: list[MeasureBoundary],
    system_members: list[list[int]],
    stem_map: dict[int, Any],
    flag_map: dict[int, int],
    config: MeasureSelectionConfig,
    rest_candidates: tuple[NoteheadCandidate, ...],
    expected_ticks: int,
) -> dict[str, list[SymbolCandidate]]:
    by_measure: dict[str, list[SymbolCandidate]] = {}
    boundaries_by_system: dict[int, list[MeasureBoundary]] = {}
    for boundary in measure_boundaries:
        boundaries_by_system.setdefault(boundary.system_index, []).append(boundary)

    assignments: list[tuple[NoteheadPseudoLabel, NoteheadCandidate, MeasureBoundary, str]] = []
    for label in labels:
        if label.index not in allowed_indices:
            continue
        notehead = noteheads[label.index]
        system_index = _system_for_staff(notehead.staff_band_index, system_members)
        matched_boundary = _boundary_for_x(boundaries_by_system.get(system_index, []), notehead.cx)
        if matched_boundary is None:
            continue
        measure_id = _measure_id(matched_boundary)
        assignments.append((label, notehead, matched_boundary, measure_id))

    note_assignment_count_by_measure: dict[str, int] = {}
    for _label, _notehead, _boundary, measure_id in assignments:
        note_assignment_count_by_measure[measure_id] = (
            note_assignment_count_by_measure.get(measure_id, 0) + 1
        )

    for label, notehead, _boundary, measure_id in assignments:
        has_stem = label.index in stem_map
        flag_count = flag_map.get(label.index, 0)
        duration_candidates = generate_duration_candidates(
            is_filled=notehead.is_filled,
            has_stem=has_stem,
            flag_count=flag_count,
            is_rest=False,
        )
        if note_assignment_count_by_measure.get(measure_id, 0) == 1:
            _add_measure_fill_duration(duration_candidates, expected_ticks)
        evidence_score = (notehead.confidence - config.threshold) * config.confidence_score_scale
        for duration in duration_candidates:
            duration.visual_score += evidence_score

        skip_penalty = -(
            config.low_confidence_skip_penalty
            + max(0.0, notehead.confidence - config.threshold) * config.high_confidence_skip_penalty
        )
        if notehead.confidence >= config.threshold:
            skip_penalty -= 1.0

        by_measure.setdefault(measure_id, []).append(
            SymbolCandidate(
                id=f"nh_{label.index}",
                bbox=notehead.bbox,
                staff_id=notehead.staff_band_index,
                measure_id=measure_id,
                x=notehead.cx,
                y=notehead.cy,
                is_filled=notehead.is_filled,
                has_stem=has_stem,
                stem_direction=getattr(stem_map.get(label.index), "direction", ""),
                flag_count=flag_count,
                is_rest=False,
                duration_candidates=duration_candidates,
                false_positive_score=skip_penalty,
                kind="note",
            )
        )

    note_candidate_count_by_measure = {
        measure_id: len(candidates) for measure_id, candidates in by_measure.items()
    }
    for rest_index, rest in enumerate(rest_candidates):
        system_index = _system_for_staff(rest.staff_band_index, system_members)
        matched_boundary = _boundary_for_x(boundaries_by_system.get(system_index, []), rest.cx)
        if matched_boundary is None:
            continue
        measure_id = _measure_id(matched_boundary)
        note_candidate_count = note_candidate_count_by_measure.get(measure_id, 0)
        rest_duration_candidates = generate_duration_candidates(
            is_filled=False,
            has_stem=False,
            flag_count=0,
            is_rest=True,
        )
        full_measure_score = 1.5 if note_candidate_count == 0 else -2.0
        rest_duration_candidates.insert(
            0,
            DurationCandidate(
                expected_ticks,
                "whole",
                stem_required=False,
                visual_score=full_measure_score,
            ),
        )
        by_measure.setdefault(measure_id, []).append(
            SymbolCandidate(
                id=f"rest_{rest_index}",
                bbox=rest.bbox,
                staff_id=rest.staff_band_index,
                measure_id=measure_id,
                x=rest.cx,
                y=rest.cy,
                is_filled=False,
                has_stem=False,
                flag_count=0,
                is_rest=True,
                duration_candidates=rest_duration_candidates,
                false_positive_score=-0.5,
                kind="rest",
            )
        )

    for candidates in by_measure.values():
        candidates.sort(key=lambda item: item.x)
    return by_measure


def _add_measure_fill_duration(
    duration_candidates: list[DurationCandidate],
    expected_ticks: int,
) -> None:
    """Allow a single ambiguous symbol to fill the detected measure duration."""
    if any(candidate.adjusted_ticks == expected_ticks for candidate in duration_candidates):
        return
    duration_candidates.insert(
        0,
        DurationCandidate(
            expected_ticks,
            "whole",
            stem_required=False,
            visual_score=0.4,
        ),
    )


def _system_for_staff(staff_index: int, system_members: list[list[int]]) -> int:
    for sys_idx, members in enumerate(system_members):
        if staff_index in members:
            return sys_idx
    return 0


def _boundary_for_x(boundaries: list[MeasureBoundary], x: float) -> MeasureBoundary | None:
    for boundary in boundaries:
        if boundary.x_start <= x < boundary.x_end:
            return boundary
    return None


def _measure_id(boundary: MeasureBoundary) -> str:
    return f"s{boundary.system_index}_m{boundary.measure_number}"
