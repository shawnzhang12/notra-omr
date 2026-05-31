"""Tests for semantic-mask to symbol-instance extraction."""

from __future__ import annotations

import numpy as np
from notra.layout.staff import StaffBand
from notra.vision.schema import SegmentationClass
from notra.vision.segmentation import SegmentationInstanceExtractor


def test_segmentation_instances_are_staff_relative() -> None:
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[18:22, 10:14] = int(SegmentationClass.NOTEHEAD_FILLED)
    mask[8:11, 40:45] = int(SegmentationClass.REST)

    band = StaffBand(line_ys=(10, 15, 20, 25, 30), interline_px=5.0)
    instances = SegmentationInstanceExtractor().extract(mask, staff_bands=[band])

    assert [item.class_name for item in instances] == ["notehead_filled", "rest"]
    assert instances[0].staff_index == 0
    assert instances[0].staff_step == 4.2
    assert instances[0].bbox.to_dict() == {"x0": 10.0, "y0": 18.0, "x1": 14.0, "y1": 22.0}


def test_segmentation_extractor_filters_small_components() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2, 2] = int(SegmentationClass.NOTEHEAD_FILLED)
    mask[10:13, 10:13] = int(SegmentationClass.NOTEHEAD_FILLED)

    instances = SegmentationInstanceExtractor().extract(mask)

    assert len(instances) == 1
    assert instances[0].mask_area == 9
