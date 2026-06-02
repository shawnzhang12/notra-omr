"""Evaluate cello staff/system/barline layout against golden MusicXML.

Usage:
  uv run python scripts/eval_cello_layout_metrics.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from notra.pipeline import stages
from notra.pipeline.config import PipelineConfig


@dataclass(frozen=True, slots=True)
class LayoutRow:
    name: str
    gt_measures: int
    pred_measures: int
    staff_count: int
    system_count: int
    per_system: list[int]


def _gt_measure_count(musicxml_path: Path) -> int:
    root = ET.parse(musicxml_path).getroot()
    return len(root.findall(".//part/measure"))


def _run_layout(image_path: Path) -> tuple[int, int, list[int]]:
    ctx = {
        "image_path": str(image_path),
        "errors": [],
        "warnings": [],
        "metrics": {},
    }
    ctx.update(PipelineConfig.cello().to_context())
    stages.load_image_stage(ctx)
    stages.detect_layout_stage(ctx)

    staff_count = len(ctx.get("staff_bands", []))
    system_members = ctx.get("system_members", [])
    barline_by_system = ctx.get("barline_by_system", {})
    per_system = [
        len(barline_by_system.get(sys_idx, []))
        for sys_idx in range(len(system_members))
    ]
    return staff_count, len(system_members), per_system


def _load_rows() -> list[LayoutRow]:
    root = Path(__file__).resolve().parents[1]
    image_root = root / "tests/fixtures/images/cello"
    golden_root = root / "tests/fixtures/golden/cello"

    rows: list[LayoutRow] = []
    for musicxml_path in sorted(golden_root.glob("*.musicxml")):
        image_path = image_root / musicxml_path.stem / "page-001.png"
        if not image_path.exists():
            continue
        staff_count, system_count, per_system = _run_layout(image_path)
        rows.append(
            LayoutRow(
                name=musicxml_path.stem,
                gt_measures=_gt_measure_count(musicxml_path),
                pred_measures=sum(per_system),
                staff_count=staff_count,
                system_count=system_count,
                per_system=per_system,
            )
        )
    return rows


def main() -> None:
    rows = _load_rows()
    if not rows:
        print("No cello fixtures found.")
        return

    print(
        f"{'Fixture':30s} {'GT':>4s} {'Pred':>5s} {'Staff':>5s} "
        f"{'Sys':>4s} Per-system"
    )
    print("-" * 88)
    for row in rows:
        print(
            f"{row.name:30s} {row.gt_measures:4d} {row.pred_measures:5d} "
            f"{row.staff_count:5d} {row.system_count:4d} {row.per_system}"
        )

    exact = sum(1 for row in rows if row.pred_measures == row.gt_measures)
    mae = mean(abs(row.pred_measures - row.gt_measures) for row in rows)
    print("\nSUMMARY")
    print(f"fixtures: {len(rows)}")
    print(f"measure-count exact: {exact}/{len(rows)}")
    print(f"measure-count MAE:   {mae:.3f}")
    gt_total = sum(row.gt_measures for row in rows)
    pred_total = sum(row.pred_measures for row in rows)
    print(f"gt/pred total:       {gt_total}/{pred_total}")


if __name__ == "__main__":
    main()
