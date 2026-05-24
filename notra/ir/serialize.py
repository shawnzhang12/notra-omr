"""Serialization helpers for Notra score IR."""

from __future__ import annotations

import json

from notra.ir.barline import Barline
from notra.ir.clef import Clef
from notra.ir.key import KeySignature
from notra.ir.measure import Measure, MeasureAttributes, Voice
from notra.ir.note import Duration, Note, Pitch
from notra.ir.rest import Rest
from notra.ir.score import Part, Score
from notra.ir.time import TimeSignature


def score_to_dict(score: Score) -> dict[str, object]:
    """Serialize score to a JSON-friendly dictionary."""
    return score.to_dict()


def score_to_json(score: Score, *, indent: int = 2) -> str:
    """Serialize score to JSON string."""
    return json.dumps(score_to_dict(score), indent=indent)


def score_from_json(raw: str) -> Score:
    """Deserialize score from JSON string."""
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON payload must be an object")
    return score_from_dict(payload)


def score_from_dict(payload: dict[str, object]) -> Score:
    """Deserialize score from dictionary payload."""
    parts_payload = _require_list(payload, "parts")
    parts = [_part_from_dict(item) for item in parts_payload]
    return Score(
        id=_require_str(payload, "id"),
        title=_require_str(payload, "title"),
        parts=parts,
    )


def _part_from_dict(payload: object) -> Part:
    part_payload = _as_dict(payload, "part")
    measures_payload = _require_list(part_payload, "measures")
    measures = [_measure_from_dict(item) for item in measures_payload]
    return Part(
        id=_require_str(part_payload, "id"),
        name=_require_str(part_payload, "name"),
        measures=measures,
    )


def _measure_from_dict(payload: object) -> Measure:
    measure_payload = _as_dict(payload, "measure")
    voices_payload = _require_list(measure_payload, "voices")
    voices = [_voice_from_dict(item) for item in voices_payload]

    attributes_value = measure_payload.get("attributes")
    attributes = _attributes_from_dict(attributes_value) if attributes_value is not None else None

    barline_value = measure_payload.get("barline")
    barline = _barline_from_dict(barline_value) if barline_value is not None else None

    return Measure(
        id=_require_str(measure_payload, "id"),
        number=_require_int(measure_payload, "number"),
        voices=voices,
        attributes=attributes,
        barline=barline,
    )


def _voice_from_dict(payload: object) -> Voice:
    voice_payload = _as_dict(payload, "voice")
    events_payload = _require_list(voice_payload, "events")
    events = [_event_from_dict(item) for item in events_payload]
    return Voice(id=_require_str(voice_payload, "id"), events=events)


def _event_from_dict(payload: object) -> Note | Rest:
    event_payload = _as_dict(payload, "event")
    kind = _require_str(event_payload, "kind")
    if kind == "note":
        return Note(
            id=_require_str(event_payload, "id"),
            pitch=_pitch_from_dict(event_payload.get("pitch")),
            duration=_duration_from_dict(event_payload.get("duration")),
            voice=_optional_int(event_payload, "voice", default=1),
            ties=tuple(_optional_str_list(event_payload, "ties")),
        )
    if kind == "rest":
        return Rest(
            id=_require_str(event_payload, "id"),
            duration=_duration_from_dict(event_payload.get("duration")),
            voice=_optional_int(event_payload, "voice", default=1),
        )
    raise ValueError(f"unknown event kind: {kind!r}")


def _attributes_from_dict(payload: object) -> MeasureAttributes:
    attr_payload = _as_dict(payload, "attributes")
    clef = (
        _clef_from_dict(attr_payload.get("clef")) if attr_payload.get("clef") is not None else None
    )
    key = _key_from_dict(attr_payload.get("key")) if attr_payload.get("key") is not None else None
    time = (
        _time_from_dict(attr_payload.get("time")) if attr_payload.get("time") is not None else None
    )
    divisions = _optional_int_or_none(attr_payload, "divisions")
    return MeasureAttributes(
        clef=clef,
        key=key,
        time=time,
        divisions=divisions,
    )


def _duration_from_dict(payload: object) -> Duration:
    duration_payload = _as_dict(payload, "duration")
    return Duration(
        numerator=_require_int(duration_payload, "numerator"),
        denominator=_require_int(duration_payload, "denominator"),
    )


def _pitch_from_dict(payload: object) -> Pitch:
    pitch_payload = _as_dict(payload, "pitch")
    step = _require_str(pitch_payload, "step")
    octave = _require_int(pitch_payload, "octave")
    alter = _optional_int(pitch_payload, "alter", default=0)
    return Pitch(step=step, octave=octave, alter=alter)


def _clef_from_dict(payload: object) -> Clef:
    clef_payload = _as_dict(payload, "clef")
    return Clef(sign=_require_str(clef_payload, "sign"), line=_require_int(clef_payload, "line"))


def _key_from_dict(payload: object) -> KeySignature:
    key_payload = _as_dict(payload, "key")
    return KeySignature(
        fifths=_require_int(key_payload, "fifths"),
        mode=_require_str(key_payload, "mode"),
    )


def _time_from_dict(payload: object) -> TimeSignature:
    time_payload = _as_dict(payload, "time")
    return TimeSignature(
        beats=_require_int(time_payload, "beats"),
        beat_type=_require_int(time_payload, "beat_type"),
    )


def _barline_from_dict(payload: object) -> Barline:
    barline_payload = _as_dict(payload, "barline")
    return Barline(style=_require_str(barline_payload, "style"))


def _as_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _ensure_dict(value: object, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")


def _require_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key!r} must be a string")
    return value


def _require_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer")
    return value


def _require_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key!r} must be a list")
    return value


def _optional_int(
    payload: dict[str, object],
    key: str,
    *,
    default: int,
) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer when provided")
    return value


def _optional_int_or_none(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer when provided")
    return value


def _optional_str_list(payload: dict[str, object], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key!r} must be a list when provided")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{key!r} must contain strings")
        result.append(item)
    return result
