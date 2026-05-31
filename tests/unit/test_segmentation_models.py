"""Tests for optional segmentation model factories."""

from __future__ import annotations

import pytest
from notra.vision.models import TinyUNetConfig, build_segmentation_model, torch_available
from notra.vision.schema import SegmentationClass, SegmentationModelConfig


def test_tiny_unet_config_matches_segmentation_labels() -> None:
    config = TinyUNetConfig(base_channels=16)
    model_config = config.to_model_config()

    assert model_config.architecture == "tiny_unet"
    assert model_config.encoder == "conv"
    assert model_config.class_count == len(SegmentationClass)


def test_build_segmentation_model_reports_missing_torch_cleanly() -> None:
    if torch_available():
        pytest.skip("missing-torch behavior only applies without torch installed")

    with pytest.raises(RuntimeError, match="requires PyTorch"):
        build_segmentation_model(TinyUNetConfig())


def test_segformer_is_deliberately_not_core() -> None:
    with pytest.raises(NotImplementedError, match="SegFormer"):
        build_segmentation_model(SegmentationModelConfig(architecture="segformer_b0"))
