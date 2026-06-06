"""Evaluate supervised measure-duration notehead selection on cello fixtures.

This is an oracle-style pseudo-label diagnostic: MusicXML duration sequences are
used during selection. Do not use this as leak-free inference accuracy.

Usage:
  uv run python scripts/eval_cello_supervised_duration_selection.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from notra.pipeline import stages
from notra.pipeline.config import PipelineConfig
from notra.semantics import expected_ticks
from notra.semantics.rhythm_solver import build_candidates_from_events
from notra.semantics.supervised_duration import select_candidates_for_duration_sequence
from notra.vision.notehead_measure_policy import _with_relaxed_rescue_candidates
from scripts.eval_cello_duration_policy import (
    _label_from_ticks,
    _parse_gt_musicxml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate supervised cello duration-constrained notehead labels."
    )
    parser.add_argument("--images-root", default="tests/fixtures/images/cello")
    parser.add_argument("--golden-root", default="tests/fixtures/golden/cello")
    parser.add_argument(
        "--output-dir",
        default="artifacts/debug/durations/cello_supervised_selection",
    )
    parser.add_argument("--include-untitled-score", action="store_true")
    parser.add_argument(
        "--no-relaxed-rescue",
        action="store_true",
        help="Disable relaxed notehead rescue; useful for coverage diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = _load_fixtures(
        Path(args.images_root),
        Path(args.golden_root),
        include_untitled=bool(args.include_untitled_score),
    )

    records: list[dict[str, Any]] = []
    for name, image_path, musicxml_path in fixtures:
        ctx = _run_candidate_pipeline(
            image_path,
            relaxed_rescue=not bool(args.no_relaxed_rescue),
        )
        gt_measures = _parse_gt_musicxml(musicxml_path)
        record = _select_page(name, image_path, musicxml_path, ctx, gt_measures)
        records.append(record)
        print(
            f"{name}: exact_measures={record['exact_measure_count']}/"
            f"{record['measure_count']} selected={record['selected_count']}/"
            f"{record['gt_event_count']} candidates={record['candidate_count']}"
        )
        for item in record["invalid_measures"]:
            print(
                f"  invalid s{item['system_index']}_m{item['measure_number']}: "
                f"{item['diagnostics']}"
            )

    summary = _summarize(records)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "mode": "supervised_musicxml_duration_sequence",
                "relaxed_rescue": not bool(args.no_relaxed_rescue),
                "excluded": [] if args.include_untitled_score else ["untitled_score"],
                "summary": summary,
                "fixtures": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"[done] wrote {summary_path}")
    return 0


def _load_fixtures(
    images_root: Path,
    golden_root: Path,
    *,
    include_untitled: bool,
) -> list[tuple[str, Path, Path]]:
    fixtures: list[tuple[str, Path, Path]] = []
    for musicxml_path in sorted(golden_root.glob("*.musicxml")):
        name = musicxml_path.stem
        if name == "untitled_score" and not include_untitled:
            continue
        image_path = images_root / name / "page-001.png"
        if image_path.exists():
            fixtures.append((name, image_path, musicxml_path))
    return fixtures


def _run_candidate_pipeline(image_path: Path, *, relaxed_rescue: bool) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "image_path": str(image_path),
        "errors": [],
        "warnings": [],
        "metrics": {},
    }
    ctx.update(PipelineConfig.cello().to_context())
    for stage_fn in (
        stages.load_image_stage,
        stages.detect_layout_stage,
        stages.classify_structure_stage,
        stages.detect_clefs_stage,
        stages.detect_noteheads_stage,
    ):
        stage_fn(ctx)

    if relaxed_rescue:
        ctx["notehead_candidates"] = list(
            _with_relaxed_rescue_candidates(
                tuple(ctx.get("notehead_candidates", [])),
                ctx,
            )
        )

    for stage_fn in (
        stages.detect_time_signature_stage,
        stages.detect_rests_stage,
        stages.detect_stems_stage,
        stages.detect_dots_stage,
        stages.detect_accidentals_stage,
        stages.assign_pitch_stage,
        stages.assemble_measures_stage,
    ):
        stage_fn(ctx)
    return ctx


def _select_page(
    name: str,
    image_path: Path,
    musicxml_path: Path,
    ctx: dict[str, Any],
    gt_measures: list[Any],
) -> dict[str, Any]:
    note_events = ctx.get("note_events", [])
    noteheads = ctx.get("notehead_candidates", [])
    time_beats = int(ctx.get("_structural_time_beats", 4) or 4)
    time_beat_type = int(ctx.get("_structural_time_beat_type", 4) or 4)
    expected = expected_ticks(time_beats, time_beat_type)
    per_measure = build_candidates_from_events(
        note_events,
        ctx.get("stem_map", {}),
        ctx.get("flag_map", {}),
        noteheads,
        ctx.get("measure_boundaries", []),
        dot_map=ctx.get("dot_map", {}),
        expected_measure_ticks=expected,
        system_members=ctx.get("system_members", []),
    )
    candidates_by_measure = {
        candidates[0].measure_id: candidates
        for candidates in per_measure
        if candidates
    }

    selected_records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    exact_count = 0
    boundaries = sorted(
        ctx.get("measure_boundaries", []),
        key=lambda item: (item.system_index, item.measure_number),
    )
    for index, (boundary, gt_measure) in enumerate(
        zip(boundaries, gt_measures, strict=False),
        start=1,
    ):
        measure_id = f"s{boundary.system_index}_m{boundary.measure_number}"
        candidates = candidates_by_measure.get(measure_id, [])
        selection = select_candidates_for_duration_sequence(
            candidates,
            tuple(gt_measure.event_ticks),
            expected_ticks=gt_measure.expected_ticks,
        )
        if selection.valid:
            exact_count += 1
        else:
            invalid.append(
                {
                    "index": index,
                    "system_index": boundary.system_index,
                    "measure_number": boundary.measure_number,
                    "candidate_count": len(candidates),
                    "target_count": len(gt_measure.event_ticks),
                    "diagnostics": list(selection.diagnostics),
                }
            )
        for target_idx, (candidate_id, ticks) in enumerate(
            zip(selection.selected_candidate_ids, gt_measure.event_ticks, strict=False)
        ):
            selected_records.append(
                {
                    "measure_id": measure_id,
                    "target_index": target_idx,
                    "candidate_id": candidate_id,
                    "duration_ticks": ticks,
                    "duration_label": _label_from_ticks(ticks),
                }
            )

    gt_event_count = sum(len(measure.event_ticks) for measure in gt_measures)
    return {
        "name": name,
        "image_path": str(image_path),
        "musicxml_path": str(musicxml_path),
        "candidate_count": sum(len(candidates) for candidates in candidates_by_measure.values()),
        "measure_count": len(gt_measures),
        "exact_measure_count": exact_count,
        "page_exact": exact_count == len(gt_measures),
        "selected_count": len(selected_records),
        "gt_event_count": gt_event_count,
        "selected_records": selected_records,
        "invalid_measures": invalid,
        "errors": list(ctx.get("errors", [])),
        "warnings": list(ctx.get("warnings", [])),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, object]:
    fixture_count = len(records)
    exact_pages = sum(int(record["page_exact"]) for record in records)
    exact_measures = sum(int(record["exact_measure_count"]) for record in records)
    total_measures = sum(int(record["measure_count"]) for record in records)
    selected_count = sum(int(record["selected_count"]) for record in records)
    gt_event_count = sum(int(record["gt_event_count"]) for record in records)
    return {
        "fixture_count": fixture_count,
        "page_exact": f"{exact_pages}/{fixture_count}",
        "exact_measures": f"{exact_measures}/{total_measures}",
        "selected_events": f"{selected_count}/{gt_event_count}",
        "all_measures_exact": exact_measures == total_measures,
    }


if __name__ == "__main__":
    raise SystemExit(main())
