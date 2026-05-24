"""Roundtrip tests for IR JSON serialization."""

from __future__ import annotations

from pathlib import Path

from notra.ir.diff import structurally_equal
from notra.ir.serialize import score_from_json, score_to_json


def test_ir_json_roundtrip_fixture() -> None:
    fixture_path = Path("tests/fixtures/scores/m001_four_quarters/expected.ir.json")
    source = fixture_path.read_text(encoding="utf-8")

    score = score_from_json(source)
    payload = score_to_json(score, indent=2)
    reparsed = score_from_json(payload)

    assert structurally_equal(score, reparsed)


def test_ir_json_roundtrip_complex_fixture() -> None:
    fixture_path = Path("tests/fixtures/scores/m010_complex_showcase/expected.ir.json")
    source = fixture_path.read_text(encoding="utf-8")

    score = score_from_json(source)
    payload = score_to_json(score, indent=2)
    reparsed = score_from_json(payload)

    assert structurally_equal(score, reparsed)
