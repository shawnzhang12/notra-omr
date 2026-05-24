"""Unit tests for id helpers."""

from __future__ import annotations

import pytest
from notra.core.ids import IdSequence, format_id, parse_id


def test_format_and_parse_id_roundtrip() -> None:
    value = format_id("measure", 7)
    assert value == "measure-007"
    assert parse_id(value) == ("measure", 7)


def test_id_sequence_is_deterministic() -> None:
    seq = IdSequence(seed={"event": 3}, width=4)

    assert seq.peek("event") == "event-0003"
    assert seq.next("event") == "event-0003"
    assert seq.next("event") == "event-0004"
    assert seq.next("measure") == "measure-0000"
    assert seq.snapshot() == {"event": 5, "measure": 1}


def test_invalid_id_inputs_raise() -> None:
    with pytest.raises(ValueError):
        format_id("Measure", 1)

    with pytest.raises(ValueError):
        format_id("measure", -1)

    with pytest.raises(ValueError):
        parse_id("measure-seven")
