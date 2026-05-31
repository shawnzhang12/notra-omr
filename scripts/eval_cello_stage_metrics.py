"""Compare cello-stage metrics between legacy and current detection modes.

Usage:
  uv run python scripts/eval_cello_stage_metrics.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from notra.pipeline import stages
from notra.pipeline.config import PipelineConfig


@dataclass(frozen=True, slots=True)
class Fixture:
    name: str
    image_path: Path
    gt_clef: str
    gt_time: str
    gt_note_count: int


def _load_fixtures() -> list[Fixture]:
    root = Path(__file__).resolve().parents[1]
    cello_dir = root / "tests/fixtures/images/cello"
    gt_dir = root / "tests/fixtures/golden/cello"

    fixtures: list[Fixture] = []
    for musicxml_path in sorted(gt_dir.glob("*.musicxml")):
        name = musicxml_path.stem
        image_path = cello_dir / name / "page-001.png"
        if not image_path.exists():
            continue

        gt_root = ET.parse(musicxml_path).getroot()
        attrs = gt_root.find(".//measure/attributes")
        gt_clef = "?"
        if attrs is not None:
            clef = attrs.find("clef")
            if clef is not None:
                sign = clef.find("sign")
                line = clef.find("line")
                if sign is not None and line is not None:
                    gt_clef = f"{sign.text}{line.text}"
        gt_time = "?"
        if attrs is not None:
            time = attrs.find("time")
            if time is not None:
                beats = time.find("beats")
                beat_type = time.find("beat-type")
                if beats is not None and beat_type is not None:
                    gt_time = f"{beats.text}/{beat_type.text}"

        gt_note_count = sum(
            1
            for note in gt_root.findall(".//note")
            if note.find("pitch") is not None or note.find("rest") is not None
        )
        fixtures.append(
            Fixture(
                name=name,
                image_path=image_path,
                gt_clef=gt_clef,
                gt_time=gt_time,
                gt_note_count=gt_note_count,
            )
        )
    return fixtures


def _run_stage_slice(
    image_path: Path,
    *,
    config: PipelineConfig | None = None,
    use_grayscale_notehead_fallback: bool | None,
    use_line_position_noteheads: bool | None,
    force_bass_clef: bool,
) -> tuple[str, str, int]:
    ctx = {
        "image_path": str(image_path),
        "errors": [],
        "warnings": [],
        "metrics": {},
        "force_bass_clef": force_bass_clef,
    }
    if config is not None:
        ctx.update(config.to_context())
    if use_grayscale_notehead_fallback is not None:
        ctx["use_grayscale_notehead_fallback"] = use_grayscale_notehead_fallback
    if use_line_position_noteheads is not None:
        ctx["use_line_position_noteheads"] = use_line_position_noteheads

    for fn in (
        stages.load_image_stage,
        stages.detect_layout_stage,
        stages.detect_clefs_stage,
        stages.detect_noteheads_stage,
        stages.detect_time_signature_stage,
    ):
        fn(ctx)

    annotations = ctx.get("staff_annotations", [])
    clef_votes = [f"{ann.clef_sign}{ann.clef_line}" for ann in annotations]
    pred_clef = Counter(clef_votes).most_common(1)[0][0] if clef_votes else "?"
    pred_time = (
        f"{ctx.get('_structural_time_beats', '?')}/"
        f"{ctx.get('_structural_time_beat_type', '?')}"
    )
    notehead_count = len(ctx.get("notehead_candidates", []))
    return pred_clef, pred_time, notehead_count


def main() -> None:
    fixtures = _load_fixtures()

    if not fixtures:
        print("No cello fixtures found.")
        return

    rows: list[dict[str, object]] = []
    for fx in fixtures:
        clef_legacy, time_legacy, heads_legacy = _run_stage_slice(
            fx.image_path,
            config=None,
            use_grayscale_notehead_fallback=True,
            use_line_position_noteheads=True,
            force_bass_clef=False,
        )
        clef_current, time_current, heads_current = _run_stage_slice(
            fx.image_path,
            config=PipelineConfig.cello(),
            use_grayscale_notehead_fallback=None,
            use_line_position_noteheads=None,
            force_bass_clef=True,
        )

        tp_legacy = min(heads_legacy, fx.gt_note_count)
        tp_current = min(heads_current, fx.gt_note_count)

        precision_legacy = tp_legacy / max(1, heads_legacy)
        precision_current = tp_current / max(1, heads_current)
        recall_legacy = tp_legacy / max(1, fx.gt_note_count)
        recall_current = tp_current / max(1, fx.gt_note_count)

        rows.append(
            {
                "name": fx.name,
                "gt_clef": fx.gt_clef,
                "gt_time": fx.gt_time,
                "legacy_clef": clef_legacy,
                "current_clef": clef_current,
                "legacy_time": time_legacy,
                "current_time": time_current,
                "gt_notes": fx.gt_note_count,
                "legacy_heads": heads_legacy,
                "current_heads": heads_current,
                "legacy_precision": precision_legacy,
                "current_precision": precision_current,
                "legacy_recall": recall_legacy,
                "current_recall": recall_current,
            }
        )

    print(
        f"{'Fixture':30s} {'ClefL':>6s} {'ClefC':>6s} {'TimeL':>6s} {'TimeC':>6s} "
        f"{'HeadsL':>7s} {'HeadsC':>7s} {'GT':>4s} {'PrecL':>6s} {'PrecC':>6s} "
        f"{'RecL':>6s} {'RecC':>6s}"
    )
    print("-" * 124)
    for row in rows:
        print(
            f"{row['name']:30s} {row['legacy_clef']:>6s} {row['current_clef']:>6s} "
            f"{row['legacy_time']:>6s} {row['current_time']:>6s} "
            f"{row['legacy_heads']:>7d} {row['current_heads']:>7d} {row['gt_notes']:>4d} "
            f"{row['legacy_precision']:.2f}  {row['current_precision']:.2f}  "
            f"{row['legacy_recall']:.2f}  {row['current_recall']:.2f}"
        )

    clef_legacy_ok = sum(1 for row in rows if row["legacy_clef"] == row["gt_clef"])
    clef_current_ok = sum(1 for row in rows if row["current_clef"] == row["gt_clef"])
    time_legacy_ok = sum(1 for row in rows if row["legacy_time"] == row["gt_time"])
    time_current_ok = sum(1 for row in rows if row["current_time"] == row["gt_time"])
    fixture_count = len(rows)

    mean_prec_legacy = mean(float(row["legacy_precision"]) for row in rows)
    mean_prec_current = mean(float(row["current_precision"]) for row in rows)
    mean_rec_legacy = mean(float(row["legacy_recall"]) for row in rows)
    mean_rec_current = mean(float(row["current_recall"]) for row in rows)
    mean_abs_err_legacy = mean(
        abs(int(row["legacy_heads"]) - int(row["gt_notes"])) for row in rows
    )
    mean_abs_err_current = mean(
        abs(int(row["current_heads"]) - int(row["gt_notes"])) for row in rows
    )

    def _f1(precision: float, recall: float) -> float:
        if (precision + recall) <= 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    mean_f1_legacy = mean(
        _f1(float(row["legacy_precision"]), float(row["legacy_recall"])) for row in rows
    )
    mean_f1_current = mean(
        _f1(float(row["current_precision"]), float(row["current_recall"])) for row in rows
    )

    print("\nSUMMARY")
    print(f"fixtures: {fixture_count}")
    print(f"clef accuracy legacy:  {clef_legacy_ok}/{fixture_count}")
    print(f"clef accuracy current: {clef_current_ok}/{fixture_count}")
    print(f"time accuracy legacy:  {time_legacy_ok}/{fixture_count}")
    print(f"time accuracy current: {time_current_ok}/{fixture_count}")
    print(f"mean precision legacy/current: {mean_prec_legacy:.3f} / {mean_prec_current:.3f}")
    print(f"mean recall legacy/current:    {mean_rec_legacy:.3f} / {mean_rec_current:.3f}")
    print(f"mean F1 legacy/current:        {mean_f1_legacy:.3f} / {mean_f1_current:.3f}")
    print(
        "mean abs count err legacy/current: "
        f"{mean_abs_err_legacy:.2f} / {mean_abs_err_current:.2f}"
    )


if __name__ == "__main__":
    main()
