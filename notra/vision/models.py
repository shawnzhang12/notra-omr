"""Optional segmentation model factories.

The core package must stay usable without PyTorch.  This module therefore keeps
all torch imports lazy: geometry, schemas, and mask extraction remain lightweight,
while training/inference code can opt into `uv run --with torch ...`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from notra.vision.schema import SegmentationClass, SegmentationModelConfig


@dataclass(frozen=True, slots=True)
class TinyUNetConfig:
    """Small U-Net config for dense OMR primitive segmentation."""

    input_channels: int = 1
    class_count: int = len(SegmentationClass)
    base_channels: int = 24
    depth: int = 4
    dropout: float = 0.0
    bilinear: bool = True

    def to_model_config(self) -> SegmentationModelConfig:
        return SegmentationModelConfig(
            architecture="tiny_unet",
            encoder="conv",
            input_channels=self.input_channels,
            class_count=self.class_count,
        )


def torch_available() -> bool:
    """Return True when PyTorch can be imported in the current environment."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def build_segmentation_model(config: TinyUNetConfig | SegmentationModelConfig | None = None) -> Any:
    """Build an optional torch segmentation model.

    `tiny_unet` is the first supported model because it is small, inspectable,
    trainable from synthetic masks, and appropriate for dense OMR primitives.
    SegFormer belongs behind a later optional dependency once the data contract
    is stable.
    """
    if config is None:
        config = TinyUNetConfig()

    if isinstance(config, SegmentationModelConfig):
        architecture = config.architecture.lower()
        if architecture in {"tiny_unet", "unet"}:
            config = TinyUNetConfig(
                input_channels=config.input_channels,
                class_count=config.class_count,
            )
        elif architecture in {"segformer", "segformer_b0", "segformer-b0"}:
            raise NotImplementedError(
                "SegFormer is intentionally not implemented in core yet; "
                "start with tiny_unet and add SegFormer behind an optional "
                "dependency after mask labels and metrics are stable."
            )
        else:
            raise ValueError(f"unknown segmentation architecture: {config.architecture}")

    return _build_tiny_unet(config)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised without torch by message test
        raise RuntimeError(
            "Segmentation model construction requires PyTorch. Use "
            "`uv run --with torch python ...` for model construction/training."
        ) from exc
    return torch, nn


def _build_tiny_unet(config: TinyUNetConfig) -> Any:
    _torch, nn = _require_torch()

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class Down(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class Up(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
            super().__init__()
            if config.bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                conv_in = in_channels + skip_channels
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                conv_in = (in_channels // 2) + skip_channels
            self.conv = DoubleConv(conv_in, out_channels)

        def forward(self, x: Any, skip: Any) -> Any:
            x = self.up(x)
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            if diff_x != 0 or diff_y != 0:
                x = nn.functional.pad(
                    x,
                    [
                        diff_x // 2,
                        diff_x - diff_x // 2,
                        diff_y // 2,
                        diff_y - diff_y // 2,
                    ],
                )
            return self.conv(_torch.cat([skip, x], dim=1))

    class TinyUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if config.depth < 2:
                raise ValueError("TinyUNet depth must be at least 2")
            channels = [config.base_channels * (2**idx) for idx in range(config.depth)]
            self.inc = DoubleConv(config.input_channels, channels[0])
            self.downs = nn.ModuleList(
                Down(channels[idx], channels[idx + 1]) for idx in range(config.depth - 1)
            )
            self.ups = nn.ModuleList(
                Up(channels[idx], channels[idx - 1], channels[idx - 1])
                for idx in range(config.depth - 1, 0, -1)
            )
            self.dropout = nn.Dropout2d(config.dropout) if config.dropout > 0 else nn.Identity()
            self.outc = nn.Conv2d(channels[0], config.class_count, kernel_size=1)
            self.config = config

        def forward(self, x: Any) -> Any:
            skips = [self.inc(x)]
            for down in self.downs:
                skips.append(down(skips[-1]))
            x = self.dropout(skips[-1])
            for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
                x = up(x, skip)
            return self.outc(x)

    return TinyUNet()
