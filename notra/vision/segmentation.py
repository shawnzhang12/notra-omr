"""Convert semantic segmentation masks into Notra symbol instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from notra.core.geometry import BBox
from notra.layout.staff import StaffBand
from notra.vision.schema import SegmentationClass, SymbolInstance


@dataclass(frozen=True, slots=True)
class SegmentationExtractionConfig:
    """Instance extraction thresholds for semantic masks."""

    min_area_by_class: dict[SegmentationClass, int] = field(default_factory=dict)
    default_min_area: int = 4

    def min_area(self, class_id: SegmentationClass) -> int:
        return self.min_area_by_class.get(class_id, self.default_min_area)


class SegmentationInstanceExtractor:
    """Extract connected-component symbol instances from a semantic mask."""

    def __init__(self, config: SegmentationExtractionConfig | None = None) -> None:
        self.config = config or SegmentationExtractionConfig()

    def extract(
        self,
        semantic_mask: np.ndarray,
        *,
        staff_bands: list[StaffBand] | None = None,
        enabled_classes: Iterable[SegmentationClass] | None = None,
        symbol_prefix: str = "seg",
    ) -> list[SymbolInstance]:
        if semantic_mask.ndim != 2:
            raise ValueError("semantic_mask must be a 2-D integer class map")

        staff_bands = staff_bands or []
        class_ids = list(enabled_classes) if enabled_classes is not None else [
            cls for cls in SegmentationClass if cls is not SegmentationClass.BACKGROUND
        ]

        instances: list[SymbolInstance] = []
        for class_id in class_ids:
            binary = semantic_mask == int(class_id)
            components = _connected_components(binary)
            for component_index, component in enumerate(components):
                area, cx, cy, x0, y0, x1, y1 = component
                if area < self.config.min_area(class_id):
                    continue

                staff_index, staff_step = _nearest_staff(cy, staff_bands)
                symbol_id = f"{symbol_prefix}_{class_id.symbol_name}_{len(instances):06d}"
                instances.append(
                    SymbolInstance(
                        symbol_id=symbol_id,
                        class_name=class_id.symbol_name,
                        bbox=BBox(float(x0), float(y0), float(x1 + 1), float(y1 + 1)),
                        center_x=cx,
                        center_y=cy,
                        staff_index=staff_index,
                        staff_step=staff_step,
                        mask_area=area,
                    )
                )

        instances.sort(
            key=lambda item: (
                item.staff_index if item.staff_index is not None else 9999,
                item.center_x,
                item.center_y,
            )
        )
        return instances


def _nearest_staff(cy: float, staff_bands: list[StaffBand]) -> tuple[int | None, float | None]:
    if not staff_bands:
        return None, None

    best_index = 0
    best_distance = float("inf")
    for idx, band in enumerate(staff_bands):
        distance = abs(cy - band.y_center)
        if distance < best_distance:
            best_index = idx
            best_distance = distance

    band = staff_bands[best_index]
    return best_index, band.staff_step_from_y(cy)


def _connected_components(
    mask: np.ndarray,
) -> list[tuple[int, float, float, int, int, int, int]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    components: list[tuple[int, float, float, int, int, int, int]] = []

    for y in range(h):
        for x in range(w):
            if not bool(mask[y, x]) or int(visited[y, x]) != 0:
                continue

            stack = [(y, x)]
            visited[y, x] = 1
            count = 0
            sum_x = 0.0
            sum_y = 0.0
            min_x = max_x = x
            min_y = max_y = y

            while stack:
                cy, cx = stack.pop()
                count += 1
                sum_x += float(cx)
                sum_y += float(cy)
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if bool(mask[ny, nx]) and int(visited[ny, nx]) == 0:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))

            components.append(
                (
                    count,
                    sum_x / float(count),
                    sum_y / float(count),
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                )
            )

    return components
