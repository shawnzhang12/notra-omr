"""Vision-layer schema and segmentation utilities."""

from notra.vision.models import TinyUNetConfig, build_segmentation_model, torch_available
from notra.vision.notehead_measure_policy import (
    LeakFreeEvaluationResult,
    MeasureSelectionConfig,
    PageMeasureSelectionResult,
    evaluate_leak_free_results,
    solve_noteheads_by_measure,
)
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
    "LeakFreeEvaluationResult",
    "MeasureSelectionConfig",
    "PageMeasureSelectionResult",
    "SelectionResult",
    "SegmentationClass",
    "SegmentationInstanceExtractor",
    "SymbolInstance",
    "ThresholdFitResult",
    "TinyUNetConfig",
    "build_segmentation_model",
    "colorize_mask",
    "evaluate_leak_free_results",
    "fit_global_threshold",
    "generate_notehead_pseudo_page",
    "generate_pseudo_segmentation_mask",
    "select_noteheads",
    "solve_noteheads_by_measure",
    "torch_available",
]
