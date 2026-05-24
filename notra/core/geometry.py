"""Geometry primitives shared across layout, IR provenance, and debugging tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """A 2D point in page coordinates."""

    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        """Serialize the point to a JSON-friendly dictionary."""
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 <= self.x0:
            raise ValueError("x1 must be greater than x0")
        if self.y1 <= self.y0:
            raise ValueError("y1 must be greater than y0")

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> "BBox":
        """Construct a bounding box from top-left plus width/height."""
        if width <= 0:
            raise ValueError("width must be greater than 0")
        if height <= 0:
            raise ValueError("height must be greater than 0")
        return cls(x0=x, y0=y, x1=x + width, y1=y + height)

    @property
    def width(self) -> float:
        """Box width."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Box height."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Box area."""
        return self.width * self.height

    @property
    def center(self) -> Point:
        """Center point of the box."""
        return Point((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def contains(self, point: Point) -> bool:
        """Return True if point lies inside the box (inclusive bounds)."""
        return self.x0 <= point.x <= self.x1 and self.y0 <= point.y <= self.y1

    def intersects(self, other: "BBox") -> bool:
        """Return True if this box overlaps another with positive area."""
        return not (
            self.x1 <= other.x0 or self.x0 >= other.x1 or self.y1 <= other.y0 or self.y0 >= other.y1
        )

    def intersection(self, other: "BBox") -> "BBox | None":
        """Return the overlap region between two boxes, if any."""
        left = max(self.x0, other.x0)
        top = max(self.y0, other.y0)
        right = min(self.x1, other.x1)
        bottom = min(self.y1, other.y1)
        if right <= left or bottom <= top:
            return None
        return BBox(left, top, right, bottom)

    def iou(self, other: "BBox") -> float:
        """Compute Intersection-over-Union with another box."""
        overlap = self.intersection(other)
        if overlap is None:
            return 0.0
        union_area = self.area + other.area - overlap.area
        if union_area <= 0:
            return 0.0
        return overlap.area / union_area

    def translate(self, dx: float, dy: float) -> "BBox":
        """Return a translated copy of this box."""
        return BBox(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def expand(self, padding: float) -> "BBox":
        """Return a box expanded uniformly in all directions."""
        if padding < 0:
            raise ValueError("padding must be non-negative")
        return BBox(self.x0 - padding, self.y0 - padding, self.x1 + padding, self.y1 + padding)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return (x, y, width, height) tuple."""
        return (self.x0, self.y0, self.width, self.height)

    def to_dict(self) -> dict[str, float]:
        """Serialize the box to a JSON-friendly dictionary."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(frozen=True, slots=True)
class PageSize:
    """Physical or pixel dimensions for one page."""

    width: float
    height: float
    unit: str = "px"

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be greater than 0")
        if self.height <= 0:
            raise ValueError("height must be greater than 0")
        if not self.unit:
            raise ValueError("unit must be non-empty")

    def to_dict(self) -> dict[str, float | str]:
        """Serialize page dimensions to a JSON-friendly dictionary."""
        return {"width": self.width, "height": self.height, "unit": self.unit}
