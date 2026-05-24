"""Stable identifier utilities for IR and pipeline artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_prefix(prefix: str) -> None:
    if not _PREFIX_RE.match(prefix):
        raise ValueError(
            "prefix must match ^[a-z][a-z0-9_]*$ (example: measure, event, symbol_node)"
        )


def format_id(prefix: str, index: int, *, width: int = 3) -> str:
    """Format a deterministic identifier like `measure-007`."""
    _validate_prefix(prefix)
    if index < 0:
        raise ValueError("index must be >= 0")
    if width < 1:
        raise ValueError("width must be >= 1")
    return f"{prefix}-{index:0{width}d}"


def parse_id(value: str) -> tuple[str, int]:
    """Parse an identifier produced by :func:`format_id`."""
    parts = value.rsplit("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"invalid id: {value!r}")
    prefix, raw_index = parts
    _validate_prefix(prefix)
    if not raw_index.isdigit():
        raise ValueError(f"invalid id index: {raw_index!r}")
    return prefix, int(raw_index)


@dataclass(slots=True)
class IdSequence:
    """Deterministic per-prefix id generator."""

    width: int = 3
    _next_by_prefix: dict[str, int] = field(default_factory=dict)

    def __init__(self, *, width: int = 3, seed: Mapping[str, int] | None = None) -> None:
        if width < 1:
            raise ValueError("width must be >= 1")
        self.width = width
        self._next_by_prefix = {}
        if seed:
            for prefix, next_index in seed.items():
                _validate_prefix(prefix)
                if next_index < 0:
                    raise ValueError("seed indexes must be >= 0")
                self._next_by_prefix[prefix] = next_index

    def peek(self, prefix: str) -> str:
        """Return the next id without incrementing the sequence."""
        _validate_prefix(prefix)
        next_index = self._next_by_prefix.get(prefix, 0)
        return format_id(prefix, next_index, width=self.width)

    def next(self, prefix: str) -> str:
        """Return the next id for a prefix and advance that prefix sequence."""
        _validate_prefix(prefix)
        next_index = self._next_by_prefix.get(prefix, 0)
        value = format_id(prefix, next_index, width=self.width)
        self._next_by_prefix[prefix] = next_index + 1
        return value

    def snapshot(self) -> dict[str, int]:
        """Return a copy of internal sequence state for serialization."""
        return dict(self._next_by_prefix)
