"""Evaluate leak-free cello duration evidence against golden MusicXML.

MusicXML is only used after inference for scoring. The pipeline chooses
durations from visual evidence plus measure-constrained decoding.

Usage:
  uv run python scripts/eval_cello_duration_policy.py
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from notra.layout.annotations import MeasureBoundary, NoteEventAnnotation
from notra.pipeline import stages
from notra.pipeline.config import PipelineConfig
from notra.semantics import (
    EIGHTH,
    HALF,
    QUARTER,
    SIXTEENTH,
    WHOLE,
    expected_ticks,
)
from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class Fixture:
    name: str
    image_path: Path
    musicxml_path: Path


@dataclass(frozen=True, slots=True)
class MeasureDurations:
    measure_number: int
    expected_ticks: int
    event_ticks: tuple[int, ...]
    labels: tuple[str, ...]

    @property
    def total_ticks(self) -> int:
        return sum(self.event_ticks)

    @property
    def valid(self) -> bool:
        return self.total_ticks == self.expected_ticks


@dataclass(frozen=True, slots=True)
class PredictedMeasureDurations(MeasureDurations):
    system_index: int


@dataclass(frozen=True, slots=True)
class FixtureDurationReport:
    name: str
    gt_event_count: int
    predicted_event_count: int
    gt_measure_count: int
    predicted_measure_count: int
    valid_measure_count: int
    total_measure_count: int
    flat_sequence_exact: bool
    measure_sequence_exact_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    invalid_measures: tuple[dict[str, object], ...]
    predicted_histogram: dict[str, int]
    gt_histogram: dict[str, int]
    predicted_dotted_histogram: dict[str, int]
    gt_dotted_histogram: dict[str, int]

    @property
    def all_measures_valid(self) -> bool:
        return self.total_measure_count > 0 and self.valid_measure_count == self.total_measure_count

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "gt_event_count": self.gt_event_count,
            "predicted_event_count": self.predicted_event_count,
            "gt_measure_count": self.gt_measure_count,
            "predicted_measure_count": self.predicted_measure_count,
            "valid_measure_count": self.valid_measure_count,
            "total_measure_count": self.total_measure_count,
            "all_measures_valid": self.all_measures_valid,
            "flat_sequence_exact": self.flat_sequence_exact,
            "measure_sequence_exact_count": self.measure_sequence_exact_count,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "invalid_measures": list(self.invalid_measures),
            "predicted_histogram": dict(self.predicted_histogram),
            "gt_histogram": dict(self.gt_histogram),
            "predicted_dotted_histogram": dict(self.predicted_dotted_histogram),
            "gt_dotted_histogram": dict(self.gt_dotted_histogram),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cello duration classification.")
    parser.add_argument("--images-root", default="tests/fixtures/images/cello")
    parser.add_argument("--golden-root", default="tests/fixtures/golden/cello")
    parser.add_argument("--output-dir", default="artifacts/debug/durations/cello")
    parser.add_argument(
        "--include-untitled-score",
        action="store_true",
        help="Include untitled_score instead of excluding it from duration validation.",
    )
    parser.add_argument("--no-overlays", action="store_true")
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

    reports: list[FixtureDurationReport] = []
    for fixture in fixtures:
        ctx = _run_duration_pipeline(fixture.image_path)
        gt = _parse_gt_musicxml(fixture.musicxml_path)
        predicted = _predicted_measures(ctx)
        report = _build_report(fixture.name, predicted, gt, ctx)
        reports.append(report)
        if not args.no_overlays and report.invalid_measures:
            _write_overlay(fixture.image_path, output_dir / f"{fixture.name}.durations.png", ctx)
        print(_format_report_row(report))

    summary = _summarize(reports)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "excluded": [] if args.include_untitled_score else ["untitled_score"],
                "summary": summary,
                "fixtures": [report.to_dict() for report in reports],
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
) -> list[Fixture]:
    fixtures: list[Fixture] = []
    for musicxml_path in sorted(golden_root.glob("*.musicxml")):
        name = musicxml_path.stem
        if name == "untitled_score" and not include_untitled:
            continue
        image_path = images_root / name / "page-001.png"
        if image_path.exists():
            fixtures.append(Fixture(name=name, image_path=image_path, musicxml_path=musicxml_path))
    return fixtures


def _run_duration_pipeline(image_path: Path) -> dict[str, Any]:
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
        stages.detect_time_signature_stage,
        stages.detect_rests_stage,
        stages.detect_stems_stage,
        stages.detect_dots_stage,
        stages.detect_accidentals_stage,
        stages.assign_pitch_stage,
        stages.assemble_measures_stage,
        stages.assign_duration_stage,
        stages.assign_voice_stage,
    ):
        try:
            stage_fn(ctx)
        except Exception as exc:  # pragma: no cover - diagnostic script path
            ctx.setdefault("errors", []).append(f"{stage_fn.__name__}: {exc}")
            break
    return ctx


def _parse_gt_musicxml(musicxml_path: Path) -> list[MeasureDurations]:
    root = ET.parse(musicxml_path).getroot()
    divisions = 1
    time_beats = 4
    time_beat_type = 4
    measures: list[MeasureDurations] = []

    for measure_index, measure in enumerate(root.findall(".//part/measure"), start=1):
        attrs = measure.find("attributes")
        if attrs is not None:
            divisions_text = attrs.findtext("divisions")
            if divisions_text:
                divisions = max(1, int(divisions_text))
            time = attrs.find("time")
            if time is not None:
                beats_text = time.findtext("beats")
                beat_type_text = time.findtext("beat-type")
                if beats_text and beat_type_text:
                    time_beats = int(beats_text)
                    time_beat_type = int(beat_type_text)

        event_ticks: list[int] = []
        labels: list[str] = []
        for note in measure.findall("note"):
            if note.find("chord") is not None:
                continue
            duration_text = note.findtext("duration")
            if not duration_text:
                continue
            ticks = int(round(int(duration_text) * QUARTER / divisions))
            dots = len(note.findall("dot"))
            note_type = note.findtext("type") or _label_from_ticks(ticks).rstrip(".")
            event_ticks.append(ticks)
            labels.append(_label(note_type, dots))

        measures.append(
            MeasureDurations(
                measure_number=measure_index,
                expected_ticks=expected_ticks(time_beats, time_beat_type),
                event_ticks=tuple(event_ticks),
                labels=tuple(labels),
            )
        )

    return measures


def _predicted_measures(ctx: dict[str, Any]) -> list[PredictedMeasureDurations]:
    events: list[NoteEventAnnotation] = ctx.get("note_events", [])
    boundaries: list[MeasureBoundary] = ctx.get("measure_boundaries", [])
    system_members: list[list[int]] = ctx.get("system_members", [])
    time_beats = int(ctx.get("_structural_time_beats", 4) or 4)
    time_beat_type = int(ctx.get("_structural_time_beat_type", 4) or 4)
    expected = expected_ticks(time_beats, time_beat_type)
    staff_to_system = _staff_to_system(system_members)

    grouped: dict[tuple[int, int], list[NoteEventAnnotation]] = {}
    for event in events:
        system_index = staff_to_system.get(event.staff_index, 0)
        boundary = _find_boundary(boundaries, system_index=system_index, x=event.cx)
        if boundary is None:
            continue
        grouped.setdefault((boundary.system_index, boundary.measure_number), []).append(event)

    predicted: list[PredictedMeasureDurations] = []
    for boundary in sorted(boundaries, key=lambda item: (item.system_index, item.measure_number)):
        measure_events = grouped.get((boundary.system_index, boundary.measure_number), [])
        measure_events.sort(key=lambda item: item.cx)
        ticks = tuple(_event_ticks(event) for event in measure_events)
        labels = tuple(_label_from_ticks(item) for item in ticks)
        predicted.append(
            PredictedMeasureDurations(
                measure_number=boundary.measure_number,
                expected_ticks=expected,
                event_ticks=ticks,
                labels=labels,
                system_index=boundary.system_index,
            )
        )
    return predicted


def _build_report(
    name: str,
    predicted: list[PredictedMeasureDurations],
    gt: list[MeasureDurations],
    ctx: dict[str, Any],
) -> FixtureDurationReport:
    predicted_labels = [label for measure in predicted for label in measure.labels]
    gt_labels = [label for measure in gt for label in measure.labels]
    paired_measure_count = min(len(predicted), len(gt))
    measure_sequence_exact = sum(
        int(predicted[index].labels == gt[index].labels)
        for index in range(paired_measure_count)
    )
    invalid = [
        {
            "system_index": measure.system_index,
            "measure_number": measure.measure_number,
            "expected_ticks": measure.expected_ticks,
            "total_ticks": measure.total_ticks,
            "labels": list(measure.labels),
        }
        for measure in predicted
        if not measure.valid
    ]
    pred_hist = Counter(predicted_labels)
    gt_hist = Counter(gt_labels)
    return FixtureDurationReport(
        name=name,
        gt_event_count=len(gt_labels),
        predicted_event_count=len(predicted_labels),
        gt_measure_count=len(gt),
        predicted_measure_count=len(predicted),
        valid_measure_count=sum(int(measure.valid) for measure in predicted),
        total_measure_count=len(predicted),
        flat_sequence_exact=tuple(predicted_labels) == tuple(gt_labels),
        measure_sequence_exact_count=measure_sequence_exact,
        errors=tuple(str(item) for item in ctx.get("errors", [])),
        warnings=tuple(str(item) for item in ctx.get("warnings", [])),
        invalid_measures=tuple(invalid),
        predicted_histogram=dict(sorted(pred_hist.items())),
        gt_histogram=dict(sorted(gt_hist.items())),
        predicted_dotted_histogram=_dotted_histogram(pred_hist),
        gt_dotted_histogram=_dotted_histogram(gt_hist),
    )


def _summarize(reports: list[FixtureDurationReport]) -> dict[str, object]:
    fixture_count = len(reports)
    total_valid = sum(report.valid_measure_count for report in reports)
    total_measures = sum(report.total_measure_count for report in reports)
    exact_pages = sum(int(report.flat_sequence_exact) for report in reports)
    total_pred_events = sum(report.predicted_event_count for report in reports)
    total_gt_events = sum(report.gt_event_count for report in reports)
    return {
        "fixture_count": fixture_count,
        "flat_sequence_exact_pages": f"{exact_pages}/{fixture_count}",
        "valid_measures": f"{total_valid}/{total_measures}",
        "valid_measure_rate": round(total_valid / max(1, total_measures), 4),
        "predicted_events": total_pred_events,
        "gt_events": total_gt_events,
        "event_count_delta": total_pred_events - total_gt_events,
        "all_measures_valid_pages": sum(int(report.all_measures_valid) for report in reports),
    }


def _write_overlay(image_path: Path, output_path: Path, ctx: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    bands = ctx.get("staff_bands", [])
    boundaries: list[MeasureBoundary] = ctx.get("measure_boundaries", [])
    predicted = _predicted_measures(ctx)
    invalid = {
        (measure.system_index, measure.measure_number): measure
        for measure in predicted
        if not measure.valid
    }
    system_members: list[list[int]] = ctx.get("system_members", [])
    for boundary in boundaries:
        key = (boundary.system_index, boundary.measure_number)
        measure = invalid.get(key)
        if measure is None:
            continue
        members = (
            system_members[boundary.system_index]
            if boundary.system_index < len(system_members)
            else []
        )
        y_values = [
            y
            for staff_index in members
            if 0 <= staff_index < len(bands)
            for y in bands[staff_index].line_ys
        ]
        if not y_values:
            y_values = [0, image.height]
        y0 = max(0, int(min(y_values) - 24))
        y1 = min(image.height, int(max(y_values) + 24))
        draw.rectangle((boundary.x_start, y0, boundary.x_end, y1), outline=(220, 40, 40), width=3)
        draw.text(
            (boundary.x_start + 4, y0 + 4),
            f"{measure.total_ticks}/{measure.expected_ticks}",
            fill=(220, 40, 40),
        )

    for dot in ctx.get("dot_candidates", []):
        x0, y0, x1, y1 = dot.bbox
        draw.ellipse((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=(0, 120, 255), width=2)
    image.save(output_path)


def _format_report_row(report: FixtureDurationReport) -> str:
    return (
        f"{report.name}: valid={report.valid_measure_count}/{report.total_measure_count} "
        f"events={report.predicted_event_count}/{report.gt_event_count} "
        f"measure_seq={report.measure_sequence_exact_count}/"
        f"{min(report.predicted_measure_count, report.gt_measure_count)} "
        f"flat_exact={report.flat_sequence_exact}"
    )


def _staff_to_system(system_members: list[list[int]]) -> dict[int, int]:
    result: dict[int, int] = {}
    for system_index, members in enumerate(system_members):
        for staff_index in members:
            result[staff_index] = system_index
    return result


def _find_boundary(
    boundaries: list[MeasureBoundary],
    *,
    system_index: int,
    x: float,
) -> MeasureBoundary | None:
    for boundary in boundaries:
        if boundary.system_index == system_index and boundary.x_start <= x < boundary.x_end:
            return boundary
    return None


def _event_ticks(event: NoteEventAnnotation) -> int:
    fraction = Fraction(event.duration_num, event.duration_den)
    return int(round(WHOLE * fraction))


def _label_from_ticks(ticks: int) -> str:
    for base_ticks, name in (
        (WHOLE, "whole"),
        (HALF, "half"),
        (QUARTER, "quarter"),
        (EIGHTH, "eighth"),
        (SIXTEENTH, "16th"),
    ):
        if ticks == base_ticks:
            return name
        if ticks == base_ticks + base_ticks // 2:
            return f"{name}."
        if ticks == base_ticks + base_ticks // 2 + base_ticks // 4:
            return f"{name}.."
    return f"unknown:{ticks}"


def _label(note_type: str, dots: int) -> str:
    normalized = {
        "16th": "16th",
        "sixteenth": "16th",
        "eighth": "eighth",
        "quarter": "quarter",
        "half": "half",
        "whole": "whole",
    }.get(note_type, note_type)
    return normalized + "." * max(0, dots)


def _dotted_histogram(histogram: Counter[str]) -> dict[str, int]:
    return {key: value for key, value in sorted(histogram.items()) if "." in key}


if __name__ == "__main__":
    raise SystemExit(main())
