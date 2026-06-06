"""Write measure-level diagnostics for the leak-free notehead solver.

This script is intentionally diagnostic, not an evaluator. It shows where the
measure-duration solver fails without using MusicXML during inference. MusicXML
is optional and only used to print page-level ground-truth notehead counts.

Example:
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/debug_leak_free_measures.py \
    --policy artifacts/training/noteheads/policy/policy.json \
    --split validation \
    --include-relaxed-rescue
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from notra.layout.staff import StaffBand
from notra.pipeline import stages
from notra.pipeline.config import PipelineConfig
from notra.vision.notehead_measure_policy import (
    MeasureSelectionConfig,
    _build_measure_boundaries,
    _measure_id,
    solve_noteheads_by_measure,
)
from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class Fixture:
    name: str
    image_path: Path
    musicxml_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug leak-free measure solver failures.")
    parser.add_argument("--images-root", default="tests/fixtures/images/cello")
    parser.add_argument("--golden-root", default="tests/fixtures/golden/cello")
    parser.add_argument("--policy", default="artifacts/training/noteheads/policy/policy.json")
    parser.add_argument("--output-dir", default="artifacts/debug/noteheads/leak_free_measures")
    parser.add_argument("--profile", default="cello", choices=["auto", "default", "cello"])
    parser.add_argument("--split", default="validation", choices=["train", "validation", "all"])
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--split-strategy", default="hash", choices=["hash", "sorted-tail"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--include-relaxed-rescue", action="store_true")
    parser.add_argument("--min-measure-candidate-confidence", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_root = Path(args.images_root)
    golden_root = Path(args.golden_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = _read_policy(Path(args.policy))
    threshold = float(
        args.threshold if args.threshold is not None else policy.get("threshold", 0.89)
    )
    include_relaxed_rescue = bool(
        args.include_relaxed_rescue or policy.get("include_relaxed_rescue", False)
    )
    fixtures = _select_fixtures(
        _load_fixtures(images_root, golden_root),
        split=str(args.split),
        validation_fraction=float(args.validation_fraction),
        strategy=str(args.split_strategy),
    )
    config = MeasureSelectionConfig(
        threshold=threshold,
        include_relaxed_rescue=include_relaxed_rescue,
        min_candidate_confidence=float(args.min_measure_candidate_confidence),
    )

    records: list[dict[str, Any]] = []
    for fixture in fixtures:
        result = solve_noteheads_by_measure(
            fixture.image_path,
            config=config,
            pipeline_config=_pipeline_config(str(args.profile), fixture.image_path),
        )
        layout = _layout_context(fixture.image_path, profile=str(args.profile))
        overlay_path = output_dir / f"{fixture.name}.measure_solver.overlay.png"
        _write_overlay(fixture.image_path, overlay_path, result, layout)
        gt_noteheads = _musicxml_pitched_notehead_count(fixture.musicxml_path)

        invalid = [item for item in result.measure_results if not item.valid]
        record = {
            "name": fixture.name,
            "image_path": str(fixture.image_path),
            "musicxml_path": str(fixture.musicxml_path) if fixture.musicxml_path else None,
            "overlay_path": str(overlay_path),
            "gt_noteheads": gt_noteheads,
            "selected_noteheads": result.selected_count,
            "candidate_count": result.candidate_count,
            "time_signature": f"{result.time_signature[0]}/{result.time_signature[1]}",
            "measure_count": result.measure_count,
            "valid_measure_count": result.valid_measure_count,
            "invalid_measure_count": len(invalid),
            "invalid_measures": [item.to_dict() for item in invalid],
            "warnings": list(result.warnings),
            "errors": list(result.errors),
        }
        records.append(record)
        print(
            f"{fixture.name}: valid={result.valid_measure_count}/{result.measure_count} "
            f"selected={result.selected_count}"
            + (f" gt={gt_noteheads}" if gt_noteheads is not None else "")
            + f" overlay={overlay_path}"
        )
        for item in invalid:
            print(
                f"  invalid {item.measure_id}: system={item.system_index + 1} "
                f"measure={item.measure_number} candidates={item.candidate_count} "
                f"selected={item.selected_count} ticks={item.total_ticks}/{item.expected_ticks} "
                f"diagnostics={list(item.diagnostics)}"
            )

    summary = {
        "schema_version": "0.1",
        "threshold": threshold,
        "include_relaxed_rescue": include_relaxed_rescue,
        "split": str(args.split),
        "record_count": len(records),
        "records": records,
    }
    summary_path = output_dir / f"{args.split}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {summary_path}")
    return 0


def _read_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pipeline_config(profile: str, image_path: Path) -> PipelineConfig:
    if profile == "default":
        return PipelineConfig.default()
    if profile == "cello":
        return PipelineConfig.cello()
    return PipelineConfig.for_image(image_path)


def _load_fixtures(images_root: Path, golden_root: Path) -> list[Fixture]:
    fixtures: list[Fixture] = []
    for image_path in sorted(images_root.glob("*/page-001.png")):
        name = image_path.parent.name
        musicxml_path = golden_root / f"{name}.musicxml"
        fixtures.append(
            Fixture(
                name=name,
                image_path=image_path,
                musicxml_path=musicxml_path if musicxml_path.exists() else None,
            )
        )
    return fixtures


def _select_fixtures(
    fixtures: list[Fixture],
    *,
    split: str,
    validation_fraction: float,
    strategy: str,
) -> list[Fixture]:
    if split == "all":
        return fixtures
    validation_count = max(1, round(len(fixtures) * validation_fraction))
    validation_count = min(validation_count, len(fixtures) - 1)
    if strategy == "sorted-tail":
        ordered = sorted(fixtures, key=lambda fixture: fixture.name)
        validation = ordered[-validation_count:]
    else:
        ordered = sorted(
            fixtures,
            key=lambda fixture: hashlib.sha256(fixture.name.encode("utf-8")).hexdigest(),
        )
        validation = ordered[:validation_count]
    validation_names = {fixture.name for fixture in validation}
    if split == "validation":
        return sorted(validation, key=lambda fixture: fixture.name)
    return [
        fixture
        for fixture in sorted(fixtures, key=lambda fixture: fixture.name)
        if fixture.name not in validation_names
    ]


def _layout_context(image_path: Path, *, profile: str) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "image_path": str(image_path),
        "errors": [],
        "warnings": [],
        "metrics": {},
    }
    ctx.update(_pipeline_config(profile, image_path).to_context())
    for stage_fn in (
        stages.load_image_stage,
        stages.detect_layout_stage,
        stages.detect_clefs_stage,
        stages.detect_noteheads_stage,
        stages.detect_time_signature_stage,
    ):
        stage_fn(ctx)
    ctx["measure_boundaries"] = _build_measure_boundaries(ctx)
    return ctx


def _write_overlay(
    image_path: Path,
    output_path: Path,
    result: Any,
    layout: dict[str, Any],
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    boundaries = {
        _measure_id(boundary): boundary for boundary in layout.get("measure_boundaries", [])
    }
    systems = _system_y_spans(layout.get("staff_bands", []), layout.get("system_members", []))
    results = {item.measure_id: item for item in result.measure_results}

    for measure_id, boundary in boundaries.items():
        item = results.get(measure_id)
        valid = bool(item.valid) if item is not None else False
        color = (42, 157, 143) if valid else (220, 40, 40)
        y0, y1 = systems.get(boundary.system_index, (0, image.height - 1))
        draw.rectangle((boundary.x_start, y0, boundary.x_end, y1), outline=color, width=4)
        label = f"S{boundary.system_index + 1} M{boundary.measure_number}"
        if item is not None:
            label += f" {item.total_ticks}/{item.expected_ticks}"
        draw.text((boundary.x_start + 4, y0 + 4), label, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _system_y_spans(
    staff_bands: list[StaffBand],
    system_members: list[list[int]],
) -> dict[int, tuple[int, int]]:
    systems = system_members or [list(range(len(staff_bands)))]
    spans: dict[int, tuple[int, int]] = {}
    for sys_idx, members in enumerate(systems):
        ys: list[float] = []
        pads: list[float] = []
        for staff_idx in members:
            if staff_idx < 0 or staff_idx >= len(staff_bands):
                continue
            band = staff_bands[staff_idx]
            ys.extend(float(y) for y in band.line_ys)
            pads.append(float(band.interline_px) * 2.0)
        if not ys:
            continue
        pad = max(pads) if pads else 20.0
        spans[sys_idx] = (max(0, int(min(ys) - pad)), int(max(ys) + pad))
    return spans


def _musicxml_pitched_notehead_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    root = ET.parse(path).getroot()
    return sum(1 for note in root.findall(".//note") if note.find("pitch") is not None)


if __name__ == "__main__":
    raise SystemExit(main())
