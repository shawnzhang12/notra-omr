"""Unit tests for duration arithmetic primitives."""

from __future__ import annotations

from fractions import Fraction

from notra.ir.note import Duration


def test_duration_fraction_is_exact() -> None:
    assert Duration(1, 4).fraction == Fraction(1, 4)
    assert Duration(3, 8).fraction == Fraction(3, 8)


def test_duration_addition_for_measure_math() -> None:
    durations = [Duration(1, 4), Duration(1, 4), Duration(1, 4), Duration(1, 4)]
    total = sum((item.fraction for item in durations), start=Fraction(0, 1))
    assert total == Fraction(1, 1)
