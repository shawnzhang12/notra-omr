"""Train and evaluate a dynamic notehead threshold policy on cello fixtures.

The fixed threshold is learned only from the training split. The dynamic mode
uses MusicXML counts as a stand-in for future measure/duration constraints;
that is an upper bound, not a deployable inference oracle.

Example:
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_notehead_threshold_policy.py \
    --images-root tests/fixtures/images/cello \
    --golden-root tests/fixtures/golden/cello \
    --output-dir artifacts/training/noteheads/policy \
    --profile cello \
    --include-relaxed-rescue
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from notra.pipeline.config import PipelineConfig
from notra.vision.notehead_policy import (
    evaluate_dynamic_target_count,
    evaluate_threshold,
    fit_global_threshold,
    threshold_grid,
)
from notra.vision.notehead_pseudolabels import (
    NoteheadPseudoLabelConfig,
    NoteheadPseudoPage,
    generate_notehead_pseudo_page,
)


@dataclass(frozen=True, slots=True)
class Fixture:
    name: str
    image_path: Path
    musicxml_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a notehead threshold policy.")
    parser.add_argument(
        "--images-root",
        default="tests/fixtures/images/cello",
        help="Root directory containing rendered page images.",
    )
    parser.add_argument(
        "--golden-root",
        default="tests/fixtures/golden/cello",
        help="MusicXML fixture root used for supervised count targets.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/training/noteheads/policy",
        help="Output directory for policy and report JSON.",
    )
    parser.add_argument(
        "--profile",
        default="cello",
        choices=["auto", "default", "cello"],
        help="Pipeline profile for candidate generation.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of fixtures reserved for validation.",
    )
    parser.add_argument(
        "--split-strategy",
        default="hash",
        choices=["hash", "sorted-tail"],
        help="Deterministic train/validation split strategy.",
    )
    parser.add_argument(
        "--include-relaxed-rescue",
        action="store_true",
        help="Union conservative candidates with relaxed grayscale/line-position rescue proposals.",
    )
    parser.add_argument("--threshold-start", type=float, default=0.40)
    parser.add_argument("--threshold-stop", type=float, default=0.98)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--min-dynamic-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence allowed for dynamic top-K repair.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_root = Path(args.images_root)
    golden_root = Path(args.golden_root)
    output_dir = Path(args.output_dir)
    fixtures = _load_fixtures(images_root, golden_root)
    if not fixtures:
        raise RuntimeError(f"no paired cello fixtures under {images_root} and {golden_root}")

    train_fixtures, validation_fixtures = _split_fixtures(
        fixtures,
        validation_fraction=float(args.validation_fraction),
        strategy=str(args.split_strategy),
    )
    config = NoteheadPseudoLabelConfig(
        include_relaxed_rescue=bool(args.include_relaxed_rescue),
        write_overlays=False,
        write_crops=False,
    )
    train_pages = [
        _generate_page(fixture, config=config, profile=str(args.profile))
        for fixture in train_fixtures
    ]
    validation_pages = [
        _generate_page(fixture, config=config, profile=str(args.profile))
        for fixture in validation_fixtures
    ]

    grid = threshold_grid(
        start=float(args.threshold_start),
        stop=float(args.threshold_stop),
        step=float(args.threshold_step),
    )
    fit = fit_global_threshold(train_pages, thresholds=grid)
    train_threshold = evaluate_threshold(train_pages, threshold=fit.threshold)
    validation_threshold = evaluate_threshold(validation_pages, threshold=fit.threshold)
    train_dynamic = evaluate_dynamic_target_count(
        train_pages,
        threshold=fit.threshold,
        min_dynamic_confidence=float(args.min_dynamic_confidence),
    )
    validation_dynamic = evaluate_dynamic_target_count(
        validation_pages,
        threshold=fit.threshold,
        min_dynamic_confidence=float(args.min_dynamic_confidence),
    )

    report: dict[str, Any] = {
        "schema_version": "0.1",
        "policy_type": "notehead_threshold_policy",
        "profile": str(args.profile),
        "include_relaxed_rescue": config.include_relaxed_rescue,
        "min_dynamic_confidence": float(args.min_dynamic_confidence),
        "threshold_grid": {
            "start": float(args.threshold_start),
            "stop": float(args.threshold_stop),
            "step": float(args.threshold_step),
        },
        "selected_threshold": fit.threshold,
        "split": {
            "strategy": str(args.split_strategy),
            "validation_fraction": float(args.validation_fraction),
            "train": [fixture.name for fixture in train_fixtures],
            "validation": [fixture.name for fixture in validation_fixtures],
        },
        "metrics": {
            "train_threshold": train_threshold.to_dict(),
            "validation_threshold": validation_threshold.to_dict(),
            "train_dynamic_target_count": train_dynamic.to_dict(),
            "validation_dynamic_target_count": validation_dynamic.to_dict(),
        },
        "pages": {
            "train": [_page_summary(page) for page in train_pages],
            "validation": [_page_summary(page) for page in validation_pages],
        },
    }
    policy = {
        "schema_version": "0.1",
        "policy_type": "notehead_threshold_policy",
        "threshold": fit.threshold,
        "include_relaxed_rescue": config.include_relaxed_rescue,
        "min_dynamic_confidence": float(args.min_dynamic_confidence),
        "fit": fit.to_dict(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    policy_path = output_dir / "policy.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _print_summary(report, report_path=report_path, policy_path=policy_path)
    return 0


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
        if musicxml_path.exists():
            fixtures.append(Fixture(name=name, image_path=image_path, musicxml_path=musicxml_path))
    return fixtures


def _split_fixtures(
    fixtures: list[Fixture],
    *,
    validation_fraction: float,
    strategy: str,
) -> tuple[list[Fixture], list[Fixture]]:
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
    train = [
        fixture
        for fixture in sorted(fixtures, key=lambda fixture: fixture.name)
        if fixture.name not in validation_names
    ]
    validation_sorted = sorted(validation, key=lambda fixture: fixture.name)
    return train, validation_sorted


def _generate_page(
    fixture: Fixture,
    *,
    config: NoteheadPseudoLabelConfig,
    profile: str,
) -> NoteheadPseudoPage:
    return generate_notehead_pseudo_page(
        fixture.image_path,
        musicxml_path=fixture.musicxml_path,
        config=config,
        pipeline_config=_pipeline_config(profile, fixture.image_path),
    )


def _page_summary(page: NoteheadPseudoPage) -> dict[str, object]:
    assert page.musicxml_counts is not None
    return {
        "name": page.image_path.parent.name,
        "gt_noteheads": page.musicxml_counts.pitched_noteheads,
        "candidate_count": len(page.labels),
        "max_possible_dynamic_count": len(page.labels),
        "feasible_by_candidate_recall": len(page.labels) >= page.musicxml_counts.pitched_noteheads,
        "label_counts": page.label_counts,
        "errors": list(page.errors),
        "warnings": list(page.warnings),
    }


def _print_summary(report: dict[str, Any], *, report_path: Path, policy_path: Path) -> None:
    metrics = report["metrics"]
    print(f"selected threshold: {report['selected_threshold']:.3f}")
    print(f"train fixtures: {len(report['split']['train'])}")
    print(f"validation fixtures: {len(report['split']['validation'])}")
    for name in (
        "train_threshold",
        "validation_threshold",
        "train_dynamic_target_count",
        "validation_dynamic_target_count",
    ):
        item = metrics[name]
        print(
            f"{name}: exact={item['exact_count']}/{item['fixture_count']} "
            f"mae={item['mean_abs_error']:.3f} "
            f"selected/target={item['total_selected']}/{item['total_target']} "
            f"infeasible={item['infeasible_count']}"
        )
    print(f"[done] wrote report to {report_path}")
    print(f"[done] wrote policy to {policy_path}")


if __name__ == "__main__":
    raise SystemExit(main())
