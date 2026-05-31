"""Regression tests for overfit beginner-cello time-signature detection."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from notra.pipeline import stages
from notra.pipeline.config import PipelineConfig


def test_cello_time_signatures_match_golden_musicxml() -> None:
    fixtures_root = Path("tests/fixtures")
    image_root = fixtures_root / "images/cello"
    golden_root = fixtures_root / "golden/cello"

    misses: list[str] = []
    checked = 0
    for musicxml_path in sorted(golden_root.glob("*.musicxml")):
        image_path = image_root / musicxml_path.stem / "page-001.png"
        if not image_path.exists():
            continue

        root = ET.parse(musicxml_path).getroot()
        time = root.find(".//measure/attributes/time")
        assert time is not None
        expected = f"{time.findtext('beats')}/{time.findtext('beat-type')}"

        ctx = {
            "image_path": str(image_path),
            "errors": [],
            "warnings": [],
            "metrics": {},
        }
        ctx.update(PipelineConfig.cello().to_context())
        for stage_fn in (
            stages.load_image_stage,
            stages.detect_layout_stage,
            stages.detect_clefs_stage,
            stages.detect_noteheads_stage,
            stages.detect_time_signature_stage,
        ):
            stage_fn(ctx)

        actual = f"{ctx.get('_structural_time_beats')}/{ctx.get('_structural_time_beat_type')}"
        checked += 1
        if actual != expected:
            misses.append(f"{musicxml_path.stem}: expected {expected}, got {actual}")

    assert checked == 18
    assert misses == []
