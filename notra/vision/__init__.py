"""Vision-layer schema and segmentation utilities."""

from notra.vision.models import TinyUNetConfig, build_segmentation_model, torch_available
from notra.vision.notehead_policy import (
    EvaluationResult,
    SelectionResult,
    ThresholdFitResult,
    fit_global_threshold,
    select_noteheads,
)
from notra.vision.notehead_pseudolabels import (
    NoteheadPseudoLabelConfig,
    NoteheadPseudoPage,
    generate_notehead_pseudo_page,
)
from notra.vision.pseudolabels import (
    PseudoMaskConfig,
    PseudoMaskResult,
    colorize_mask,
    generate_pseudo_segmentation_mask,
)
from notra.vision.schema import SegmentationClass, SymbolInstance
from notra.vision.segmentation import SegmentationInstanceExtractor

__all__ = [
    "PseudoMaskConfig",
    "PseudoMaskResult",
    "NoteheadPseudoLabelConfig",
    "NoteheadPseudoPage",
    "EvaluationResult",
    "SelectionResult",
    "SegmentationClass",
    "SegmentationInstanceExtractor",
    "SymbolInstance",
    "ThresholdFitResult",
    "TinyUNetConfig",
    "build_segmentation_model",
    "colorize_mask",
    "fit_global_threshold",
    "generate_notehead_pseudo_page",
    "generate_pseudo_segmentation_mask",
    "select_noteheads",
    "torch_available",
]
