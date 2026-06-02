"""Regression tests for rendered cello staff and measure-barline detection."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from notra.pipeline import stages
from notra.pipeline.config import PipelineConfig


def test_cello_measure_barlines_match_golden_musicxml_counts() -> None:
    fixtures_root = Path("tests/fixtures")
    image_root = fixtures_root / "images/cello"
    golden_root = fixtures_root / "golden/cello"

    misses: list[str] = []
    checked = 0
    for musicxml_path in sorted(golden_root.glob("*.musicxml")):
        image_path = image_root / musicxml_path.stem / "page-001.png"
        if not image_path.exists():
            continue

        expected = len(ET.parse(musicxml_path).getroot().findall(".//part/measure"))
        ctx = {
            "image_path": str(image_path),
            "errors": [],
            "warnings": [],
            "metrics": {},
        }
        ctx.update(PipelineConfig.cello().to_context())
        stages.load_image_stage(ctx)
        stages.detect_layout_stage(ctx)

        barline_by_system = ctx.get("barline_by_system", {})
        actual = sum(len(xs) for xs in barline_by_system.values())
        checked += 1
        if actual != expected:
            misses.append(f"{musicxml_path.stem}: expected {expected}, got {actual}")

    assert checked == 18
    assert misses == []
