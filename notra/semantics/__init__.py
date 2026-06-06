"""
Duration lattice: data structures for candidate generation and constrained decoding.

Design principle: visual model proposes, measure solver disposes.
Never commit to a scalar duration from local evidence alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------------------
# Tick constants (1920 ticks per whole note = divisible by 2,3,4,5,6,8,10,12)
# ---------------------------------------------------------------------------
WHOLE = 1920
HALF = 960
QUARTER = 480
EIGHTH = 240
SIXTEENTH = 120
THIRTY_SECOND = 60
SIXTY_FOURTH = 30

DURATION_NAME_TO_TICKS: dict[str, int] = {
    "whole": WHOLE,
    "half": HALF,
    "quarter": QUARTER,
    "eighth": EIGHTH,
    "16th": SIXTEENTH,
    "32nd": THIRTY_SECOND,
    "64th": SIXTY_FOURTH,
}

TICKS_TO_NAME: dict[int, str] = {v: k for k, v in DURATION_NAME_TO_TICKS.items()}


def expected_ticks(time_beats: int, time_beat_type: int) -> int:
    """Compute integer ticks per measure from time signature."""
    # beats * (WHOLE / beat_type)
    # e.g. 4/4 → 4 * 1920 / 4 = 1920
    #      6/8 → 6 * 1920 / 8 = 1440
    #      3/4 → 3 * 1920 / 4 = 1440
    return time_beats * WHOLE // time_beat_type


# ---------------------------------------------------------------------------
# Duration candidate
# ---------------------------------------------------------------------------


@dataclass
class DurationCandidate:
    """One duration hypothesis for a symbol."""

    duration_ticks: int
    note_type: str  # "whole", "half", "quarter", "eighth", "16th", "32nd", "64th"
    dots: int = 0
    beams: int = 0
    flags: int = 0
    stem_required: bool = True
    visual_score: float = 0.0
    grammar_score: float = 0.0
    evidence: str = ""

    @property
    def adjusted_ticks(self) -> int:
        """Duration including dots."""
        if self.dots == 0:
            return self.duration_ticks
        base = self.duration_ticks
        if self.dots == 1:
            return base + base // 2
        if self.dots == 2:
            return base + base // 2 + base // 4
        return base


# ---------------------------------------------------------------------------
# Symbol candidate
# ---------------------------------------------------------------------------


@dataclass
class SymbolCandidate:
    """One detected symbol with multiple duration hypotheses."""

    id: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    staff_id: int
    measure_id: str = ""
    x: float = 0.0
    y: float = 0.0
    is_filled: bool = False
    has_stem: bool = False
    stem_direction: str = ""  # "up", "down", ""
    flag_count: int = 0
    dot_count: int = 0
    is_rest: bool = False
    duration_candidates: list[DurationCandidate] = field(default_factory=list)
    false_positive_score: float = -4.0  # penalty for skipping this candidate
    kind: str = "note"  # "note" or "rest"


# ---------------------------------------------------------------------------
# Measure decode result
# ---------------------------------------------------------------------------


@dataclass
class SelectedEvent:
    """One event selected by the decoder."""

    candidate_id: str
    duration_ticks: int
    note_type: str
    dots: int
    voice: int
    is_rest: bool
    score: float


@dataclass
class VoiceDecode:
    """Decoded events for one voice in a measure."""

    voice: int
    events: list[SelectedEvent]
    total_ticks: int
    is_valid: bool


@dataclass
class MeasureDecode:
    """Result of measure rhythm decoding."""

    measure_id: str
    expected_ticks: int
    voices: list[VoiceDecode]
    selected_events: list[SelectedEvent]
    rejected_candidates: list[str]  # candidate IDs marked as false positive
    total_score: float
    valid: bool
    diagnostics: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Candidate generation from pipeline note events
# ---------------------------------------------------------------------------


def generate_duration_candidates(
    is_filled: bool,
    has_stem: bool,
    flag_count: int = 0,
    dot_count: int = 0,
    is_rest: bool = False,
) -> list[DurationCandidate]:
    """Generate duration hypotheses from visual evidence.

    Returns candidates sorted by visual_score (best first).
    False positive is always included as the last option.
    """
    candidates: list[DurationCandidate] = []

    dot_count = max(0, min(2, int(dot_count)))

    if is_rest:
        # Rest candidates: mostly quarter, with alternatives
        candidates = [
            DurationCandidate(
                QUARTER, "quarter", stem_required=False, visual_score=-0.5,
                evidence="rest_shape",
            ),
            DurationCandidate(
                EIGHTH, "eighth", stem_required=False, visual_score=-1.0,
                evidence="rest_shape_alt",
            ),
            DurationCandidate(
                HALF, "half", stem_required=False, visual_score=-1.0,
                evidence="rest_shape_alt",
            ),
            DurationCandidate(
                WHOLE, "whole", stem_required=False, visual_score=-1.5,
                evidence="rest_shape_alt",
            ),
            DurationCandidate(
                SIXTEENTH, "16th", stem_required=False, visual_score=-1.5,
                evidence="rest_shape_alt",
            ),
        ]
    elif not is_filled:
        # Open notehead
        if has_stem:
            candidates = [
                DurationCandidate(
                    HALF, "half", stem_required=True, visual_score=-0.1,
                    evidence="open_head_stem",
                ),
                DurationCandidate(
                    QUARTER, "quarter", stem_required=True, visual_score=-1.8,
                    evidence="open_head_stem_fallback",
                ),
                DurationCandidate(
                    WHOLE, "whole", stem_required=False, visual_score=-2.5,
                    evidence="open_head_stem_fallback",
                ),
            ]
        else:
            # Open + no stem: strongly prefer whole note
            candidates = [
                DurationCandidate(
                    WHOLE, "whole", stem_required=False, visual_score=-0.2,
                    evidence="open_head_no_stem",
                ),
                DurationCandidate(
                    HALF, "half", stem_required=True, visual_score=-3.0,
                    evidence="open_head_no_stem_fallback",
                ),
            ]
    else:
        # Filled notehead (with reliable stems from global detector)
        if has_stem:
            if flag_count == 1:
                candidates = [
                    DurationCandidate(
                        EIGHTH, "eighth", flags=1, visual_score=-0.1,
                        evidence="filled_head_stem_one_beam_or_flag",
                    ),
                    DurationCandidate(
                        QUARTER, "quarter", flags=0, visual_score=-0.8,
                        evidence="filled_head_stem_one_beam_or_flag_fallback",
                    ),
                    DurationCandidate(
                        SIXTEENTH, "16th", flags=2, visual_score=-1.1,
                        evidence="filled_head_stem_one_beam_or_flag_fallback",
                    ),
                ]
            elif flag_count == 2:
                candidates = [
                    DurationCandidate(
                        SIXTEENTH, "16th", flags=2, visual_score=-0.1,
                        evidence="filled_head_stem_two_beams_or_flags",
                    ),
                    DurationCandidate(
                        EIGHTH, "eighth", flags=1, visual_score=-0.7,
                        evidence="filled_head_stem_two_beams_or_flags_fallback",
                    ),
                    DurationCandidate(
                        QUARTER, "quarter", flags=0, visual_score=-1.2,
                        evidence="filled_head_stem_two_beams_or_flags_fallback",
                    ),
                ]
            elif flag_count >= 3:
                candidates = [
                    DurationCandidate(
                        THIRTY_SECOND, "32nd", flags=3, visual_score=-0.1,
                        evidence="filled_head_stem_three_beams_or_flags",
                    ),
                    DurationCandidate(
                        SIXTEENTH, "16th", flags=2, visual_score=-1.0,
                        evidence="filled_head_stem_three_beams_or_flags_fallback",
                    ),
                ]
            else:
                # Stem but no flag/beam: quarter or inferred eighth only.
                # No sixteenth without beam evidence.
                candidates = [
                    DurationCandidate(
                        QUARTER, "quarter", visual_score=-0.1,
                        evidence="filled_head_stem_no_beam_or_flag",
                    ),
                    DurationCandidate(
                        EIGHTH, "eighth", beams=1, visual_score=-0.5,
                        evidence="filled_head_stem_no_beam_or_flag_fallback",
                    ),
                ]
        else:
            # Filled but no stem: could be quarter with missed stem, or fragment
            candidates = [
                DurationCandidate(
                    QUARTER, "quarter", stem_required=True, visual_score=-0.7,
                    evidence="filled_head_missing_stem",
                ),
                DurationCandidate(
                    EIGHTH, "eighth", stem_required=True, visual_score=-1.2,
                    evidence="filled_head_missing_stem_fallback",
                ),
                DurationCandidate(
                    HALF, "half", stem_required=True, visual_score=-2.0,
                    evidence="filled_head_missing_stem_fallback",
                ),
            ]

    if dot_count <= 0:
        return candidates

    dotted: list[DurationCandidate] = []
    undotted: list[DurationCandidate] = []
    for candidate in candidates:
        dotted.append(
            replace(
                candidate,
                dots=dot_count,
                visual_score=candidate.visual_score + 0.35,
                evidence=f"{candidate.evidence}+augmentation_dot",
            )
        )
        undotted.append(
            replace(
                candidate,
                visual_score=candidate.visual_score - 1.20,
                evidence=f"{candidate.evidence}+dot_rejected_fallback",
            )
        )

    return dotted + undotted
