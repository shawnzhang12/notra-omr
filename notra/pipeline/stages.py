"""OMR pipeline stages: detection → classification → assembly → export.

Each stage takes a shared context dict and mutates it, following a
functional-core/imperative-shell pattern. Stages can be composed into
a full recognition pipeline.

Stage ordering:
     1. load_image          — read + binarize the input image
     2. detect_layout       — staff lines, barlines, systems
     3. detect_clefs        — clef classification per staff
     4. detect_noteheads    — notehead candidate extraction
     5. detect_time         — classify simple opening time signatures
     6. detect_rests        — rest symbol detection
     7. detect_stems        — stem detection per notehead
     8. detect_accidentals  — accidental detection per notehead
     9. assign_pitch        — map staff position → diatonic pitch, apply accidentals
    10. assign_duration     — classify notehead type → duration with flags
    11. assign_voice        — split noteheads into voices via stem direction
    12. assemble_measures   — group events by per-system barline boundaries
    13. build_score         — construct notra.ir.Score with multi-voice parts
    14. export_musicxml     — serialize to MusicXML string
"""

from __future__ import annotations

from typing import Any

import numpy as np
from notra.core.geometry import BBox
from notra.ir.clef import Clef
from notra.ir.key import KeySignature
from notra.ir.measure import Measure, MeasureAttributes, Voice
from notra.ir.note import Duration, Note, Pitch
from notra.ir.rest import Rest
from notra.ir.score import Part, Score
from notra.ir.time import TimeSignature
from notra.layout.annotations import (
    MeasureBoundary,
    NoteEventAnnotation,
    PageAnnotations,
    PipelineResult,
    StaffAnnotation,
)
from notra.layout.measure import detect_measure_barlines, estimate_staff_x_extent
from notra.layout.staff import (
    StaffBand,
    detect_staff_bands_from_horizontal_runs,
    detect_staff_lines,
    group_staff_bands,
    staff_step_to_pitch,
)
from notra.layout.symbol import (
    NoteheadCandidate,
    StemCandidate,
    assign_voice_from_stems,
    classify_duration,
    detect_accidentals,
    detect_barlines,
    detect_clef_region,
    detect_noteheads,
    detect_rests,
)
from notra.pipeline.config import PipelineConfig
from PIL import Image

# ---------------------------------------------------------------------------
# Stage 1: Image loading
# ---------------------------------------------------------------------------


def load_image_stage(ctx: dict[str, Any]) -> None:
    """Load and binarize the input image.

    Handles:
      - Standard grayscale/RGB images via convert("L").
      - RGBA images with content in the alpha channel (common from Verovio
        renders) — extracts alpha as ink (255=ink, 0=background).
      - RGBA images with content in RGB channels — converts to L as usual.
    """
    image_path = ctx["image_path"]
    img = Image.open(image_path)

    if img.mode == "RGBA":
        arr = np.asarray(img)
        alpha = arr[:, :, 3]
        rgb_max = arr[:, :, :3].max()

        if rgb_max <= 2 and alpha.max() >= 200:
            gray = alpha.astype(np.uint8)
        elif alpha.min() >= 200:
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
        else:
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
    else:
        img_l = img.convert("L")
        gray = np.asarray(img_l, dtype=np.uint8)

    # Auto-upscale if requested (helps dense scores with thin staff lines)
    upscale_factor = ctx.get("upscale_factor", 0)
    if upscale_factor > 1:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(gray)
        new_w = int(pil_img.width * upscale_factor)
        new_h = int(pil_img.height * upscale_factor)
        pil_img = pil_img.resize((new_w, new_h), PILImage.NEAREST)
        gray = np.asarray(pil_img, dtype=np.uint8)

    ctx["gray"] = gray
    ctx["image_width"] = gray.shape[1]
    ctx["image_height"] = gray.shape[0]


# ---------------------------------------------------------------------------
# Stage 2: Layout detection
# ---------------------------------------------------------------------------


def detect_layout_stage(ctx: dict[str, Any]) -> None:
    """Detect staff lines, barlines, and staff bands.

    If staff detection fails on the original image, automatically upscales
    2x and retries (helps with dense orchestral scores).
    """
    gray = ctx["gray"]

    staff_line_ys = detect_staff_lines(gray, threshold_sigma=1.0)
    bands, interline = group_staff_bands(staff_line_ys)

    if ctx.get("profile_name") == "cello":
        rendered_bands = detect_staff_bands_from_horizontal_runs(gray)
        if len(rendered_bands) >= len(bands):
            bands = rendered_bands
            interline = float(np.median([band.interline_px for band in bands]))

    # Auto-upscale retry for dense scores (2x then 3x)
    for upscale in [2, 3]:
        if bands or ctx.get("_upscale_retried"):
            break
        ctx["_upscale_retried"] = True
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(gray)
        new_w = pil_img.width * upscale
        new_h = pil_img.height * upscale
        pil_img = pil_img.resize((new_w, new_h), PILImage.NEAREST)
        gray_up = np.asarray(pil_img, dtype=np.uint8)
        ctx["gray"] = gray_up
        ctx["image_width"] = gray_up.shape[1]
        ctx["image_height"] = gray_up.shape[0]
        gray = gray_up
        staff_line_ys = detect_staff_lines(gray_up, threshold_sigma=0.7)
        bands, interline = group_staff_bands(staff_line_ys)

    if ctx.get("profile_name") == "cello":
        rendered_bands = detect_staff_bands_from_horizontal_runs(gray)
        if len(rendered_bands) >= len(bands):
            bands = rendered_bands
            interline = float(np.median([band.interline_px for band in bands]))

    if not bands:
        ctx["errors"].append("No staff bands detected")
        return

    ink = _ensure_ink(ctx)
    system_members: list[list[int]] = _detect_system_members(bands)

    if ctx.get("profile_name") == "cello":
        staff_barlines = detect_measure_barlines(ink, gray, bands)
        barline_by_system = _barlines_by_system_from_staff_barlines(
            staff_barlines,
            system_members,
            bands,
        )
        barline_xs = sorted({x for values in barline_by_system.values() for x in values})
    else:
        barline_xs = detect_barlines(ink, bands, min_length_ratio=4.0)
        if len(barline_xs) < max(2, len(bands)):
            denser = detect_barlines(ink, bands, min_length_ratio=3.5)
            if len(denser) > len(barline_xs):
                barline_xs = denser
        barline_by_system = _assign_global_barlines_to_systems(
            ink,
            bands,
            system_members,
            barline_xs,
        )

    ctx["staff_bands"] = bands
    ctx["interline_px"] = interline
    ctx["barline_xs"] = barline_xs
    ctx["barline_by_system"] = barline_by_system
    ctx["system_members"] = system_members
    ctx["ink"] = ink


# ---------------------------------------------------------------------------
# Stage 3: Clef detection
# ---------------------------------------------------------------------------


def detect_clefs_stage(ctx: dict[str, Any]) -> None:
    """Detect the clef for each staff band.

    When structural classifier predictions are available, uses those
    instead of deterministic detection (far more accurate).
    """
    ink = _ensure_ink(ctx)
    bands: list[StaffBand] = ctx["staff_bands"]
    barline_xs: list[float] = ctx.get("barline_xs", [])

    # Use classifier clefs if available
    classifier_clefs: list[tuple[str, int]] | None = ctx.get("_structural_clefs")
    force_bass_clef = bool(ctx.get("force_bass_clef", False))

    staff_annotations: list[StaffAnnotation] = []
    for idx, band in enumerate(bands):
        if classifier_clefs and idx < len(classifier_clefs):
            clef_sign, clef_line = classifier_clefs[idx]
        else:
            clef_sign, clef_line = detect_clef_region(ink, band, barline_xs=barline_xs)
        if force_bass_clef:
            clef_sign, clef_line = ("F", 4)

        # Detect key signature
        from notra.layout.symbol import detect_key_signature
        key_fifths = detect_key_signature(ink, band, clef_sign, barline_xs=barline_xs)

        staff_annotations.append(
            StaffAnnotation(
                staff_index=idx,
                band=band,
                clef_sign=clef_sign,
                clef_line=clef_line,
                key_fifths=key_fifths,
            )
        )

    ctx["staff_annotations"] = staff_annotations


# ---------------------------------------------------------------------------
# Stage 4: Notehead detection
# ---------------------------------------------------------------------------


def detect_noteheads_stage(ctx: dict[str, Any]) -> None:
    """Detect notehead candidates from the ink image."""
    ink = _ensure_ink(ctx)
    bands: list[StaffBand] = ctx["staff_bands"]
    noteheads = detect_noteheads(
        ink,
        bands,
        gray=ctx.get("gray"),
        use_grayscale_fallback=bool(ctx.get("use_grayscale_notehead_fallback", False)),
        use_line_position_pass=bool(ctx.get("use_line_position_noteheads", False)),
    )

    # Profile-controlled rescue: if conservative pass under-detects, allow a
    # bounded grayscale fallback expansion.
    if bool(ctx.get("low_density_grayscale_rescue", False)) and noteheads:
        density = len(noteheads) / float(max(1, len(bands)))
        density_threshold = float(ctx.get("cello_low_density_threshold", 22.0))
        growth_cap = float(ctx.get("cello_gray_growth_cap", 1.5))
        if density < density_threshold:
            gray_rescue = detect_noteheads(
                ink,
                bands,
                gray=ctx.get("gray"),
                use_grayscale_fallback=True,
                use_line_position_pass=False,
            )
            max_rescue_count = int(round(len(noteheads) * growth_cap))
            if len(noteheads) < len(gray_rescue) <= max_rescue_count:
                ctx.setdefault("warnings", []).append(
                    "notehead density low; enabled bounded grayscale rescue"
                )
                noteheads = gray_rescue

    # Note: size-consistency filter is applied inside detect_noteheads
    # via median-area analysis (see symbol.py:216-229)
    ctx["notehead_candidates"] = noteheads


# ---------------------------------------------------------------------------
# Stage 5: Time signature detection
# ---------------------------------------------------------------------------


def detect_time_signature_stage(ctx: dict[str, Any]) -> None:
    """Detect simple opening time signatures unless structural predictions exist."""
    if "_structural_time_beats" in ctx and "_structural_time_beat_type" in ctx:
        return

    ink = _ensure_ink(ctx)
    bands: list[StaffBand] = ctx["staff_bands"]

    from notra.layout.time_signature import detect_time_signature

    candidate = detect_time_signature(ink, bands)
    ctx["_structural_time_beats"] = candidate.beats
    ctx["_structural_time_beat_type"] = candidate.beat_type
    ctx["time_signature_candidate"] = candidate
    ctx.setdefault("metrics", {})["time_signature"] = candidate.signature
    ctx.setdefault("metrics", {})["time_signature_visual_class"] = candidate.visual_class


# ---------------------------------------------------------------------------
# Stage 6: Rest detection
# ---------------------------------------------------------------------------


def detect_rests_stage(ctx: dict[str, Any]) -> None:
    """Detect rest symbols in staff bands, filtered by notehead proximity."""
    ink = _ensure_ink(ctx)
    bands: list[StaffBand] = ctx["staff_bands"]
    noteheads: list[NoteheadCandidate] = ctx.get("notehead_candidates", [])

    rests = detect_rests(ink, bands, noteheads=noteheads)
    ctx["rest_candidates"] = rests


# ---------------------------------------------------------------------------
# Stage 7: Stem detection
# ---------------------------------------------------------------------------


def detect_stems_stage(ctx: dict[str, Any]) -> None:
    """Detect stems globally via column-run detection, then attach to noteheads."""
    ink = _ensure_ink(ctx)
    noteheads: list[NoteheadCandidate] = ctx.get("notehead_candidates", [])
    bands: list[StaffBand] = ctx["staff_bands"]

    if not noteheads:
        ctx["stem_map"] = {}
        ctx["flag_map"] = {}
        return

    from notra.layout.stem_detector import detect_stems_global
    global_stems, stem_map_raw = detect_stems_global(ink, bands, noteheads)

    # Convert to pipeline format
    pipeline_stem_map: dict[int, Any] = {}
    for nh_idx, gs in stem_map_raw.items():
        pipeline_stem_map[nh_idx] = StemCandidate(
            notehead_cx=noteheads[nh_idx].cx,
            notehead_cy=noteheads[nh_idx].cy,
            direction=gs.direction,
            top_y=gs.y0 if gs.direction == "up" else gs.y1,
            bottom_y=gs.y1 if gs.direction == "up" else gs.y0,
            center_x=gs.x_center,
            length_px=gs.height,
        )

    ctx["stem_map"] = pipeline_stem_map

    # --- Beam detection ---
    from notra.layout.beam_detector import detect_beams
    interline = float(np.median([b.interline_px for b in bands]))
    beams = detect_beams(ink, bands, list(stem_map_raw.values()), interline)
    ctx["beam_candidates"] = beams

    # Build flag_map from beam connections
    # Map beam detector stem index → pipeline notehead index
    flag_map: dict[int, int] = {}
    stem_list = list(stem_map_raw.values())
    stem_idx_to_nh: dict[int, int] = {}
    for si, stem in enumerate(stem_list):
        # Find which notehead this stem is attached to
        for nh_idx, gs in stem_map_raw.items():
            if gs is stem:
                stem_idx_to_nh[si] = nh_idx
                break

    for beam in beams:
        for si in beam.connected_stems:
            nh_idx = stem_idx_to_nh.get(si)
            if nh_idx is not None:
                flag_map[nh_idx] = max(flag_map.get(nh_idx, 0), beam.level)

    ctx["flag_map"] = flag_map
    ctx.setdefault("metrics", {})["global_stems_found"] = len(global_stems)
    ctx.setdefault("metrics", {})["stems_attached"] = len(stem_map_raw)
    ctx.setdefault("metrics", {})["beams_found"] = len(beams)
    ctx.setdefault("metrics", {})["beamed_notes"] = len(flag_map)


# ---------------------------------------------------------------------------
# Stage 8: Accidental detection
# ---------------------------------------------------------------------------


def detect_accidentals_stage(ctx: dict[str, Any]) -> None:
    """Detect accidentals left of noteheads."""
    ink = _ensure_ink(ctx)
    noteheads: list[NoteheadCandidate] = ctx.get("notehead_candidates", [])
    bands: list[StaffBand] = ctx["staff_bands"]

    if not noteheads:
        ctx["accidental_map"] = {}
        return

    accidental_map = detect_accidentals(ink, noteheads, bands)
    ctx["accidental_map"] = accidental_map


# ---------------------------------------------------------------------------
# Stage 9: Pitch assignment
# ---------------------------------------------------------------------------


def assign_pitch_stage(ctx: dict[str, Any]) -> None:
    """Assign diatonic pitch to each notehead, applying detected accidentals."""
    noteheads: list[NoteheadCandidate] = ctx.get("notehead_candidates", [])
    staff_anns: list[StaffAnnotation] = ctx.get("staff_annotations", [])
    bands: list[StaffBand] = ctx["staff_bands"]
    accidental_map: dict[int, int] = ctx.get("accidental_map", {})

    note_events: list[NoteEventAnnotation] = []
    for idx, nh in enumerate(noteheads):
        staff_idx = getattr(nh, 'staff_band_index', 0)
        if staff_idx >= len(staff_anns) or staff_idx >= len(bands):
            continue

        sa = staff_anns[staff_idx] if staff_idx < len(staff_anns) else None
        clef_sign = sa.clef_sign if sa else "G"
        clef_line = sa.clef_line if sa else 2

        step_name, octave, alter = staff_step_to_pitch(
            nh.staff_step, clef_sign, clef_line
        )

        # Apply detected accidental (overrides diatonic alter=0)
        detected_alter = accidental_map.get(idx)
        if detected_alter is not None:
            alter = detected_alter

        note_events.append(
            NoteEventAnnotation(
                event_index=idx,
                staff_index=staff_idx,
                staff_step=nh.staff_step,
                diatonic_step=step_name,
                octave=octave,
                alter=alter,
                duration_num=1,
                duration_den=4,
                cx=nh.cx,
                cy=nh.cy,
                bbox=BBox(
                    float(nh.bbox[0]), float(nh.bbox[1]),
                    float(nh.bbox[2]), float(nh.bbox[3]),
                ),
            )
        )

    # Add rests as note events with is_rest=True
    rest_candidates: list[NoteheadCandidate] = ctx.get("rest_candidates", [])
    for rest_idx, rst in enumerate(rest_candidates):
        staff_idx = getattr(rst, 'staff_band_index', 0)
        if staff_idx >= len(staff_anns) or staff_idx >= len(bands):
            continue

        sa = staff_anns[staff_idx] if staff_idx < len(staff_anns) else None
        clef_sign = sa.clef_sign if sa else "G"
        clef_line = sa.clef_line if sa else 2
        step_name, octave, alter = staff_step_to_pitch(
            rst.staff_step, clef_sign, clef_line
        )

        note_events.append(
            NoteEventAnnotation(
                event_index=len(noteheads) + rest_idx,
                staff_index=staff_idx,
                staff_step=rst.staff_step,
                diatonic_step=step_name,
                octave=octave,
                alter=alter,
                duration_num=1,
                duration_den=4,
                is_rest=True,
                cx=rst.cx,
                cy=rst.cy,
                bbox=BBox(
                    float(rst.bbox[0]), float(rst.bbox[1]),
                    float(rst.bbox[2]), float(rst.bbox[3]),
                ),
            )
        )

    ctx["note_events"] = note_events


# ---------------------------------------------------------------------------
# Stage 9: Duration assignment
# ---------------------------------------------------------------------------


def assign_duration_stage(ctx: dict[str, Any]) -> None:
    """Assign durations via measure-constrained rhythm solver.

    Instead of committing to a scalar duration per notehead, generates
    multiple duration hypotheses and uses DP to find the best legal
    assignment under the time signature constraint.
    """
    noteheads: list[NoteheadCandidate] = ctx.get("notehead_candidates", [])
    note_events: list[NoteEventAnnotation] = ctx.get("note_events", [])
    stem_map: dict[int, Any] = ctx.get("stem_map", {})
    flag_map: dict[int, int] = ctx.get("flag_map", {})
    measure_boundaries = ctx.get("measure_boundaries", [])
    barline_xs: list[float] = ctx.get("barline_xs", [])
    time_beats = ctx.get("_structural_time_beats", 4)
    time_beat_type = ctx.get("_structural_time_beat_type", 4)

    if not note_events:
        return

    # If barlines are missing, rhythm-constrained decoding becomes
    # over-aggressive (one giant pseudo-measure) and rejects most symbols.
    # Fall back to per-note geometry duration classification.
    if not measure_boundaries or not barline_xs:
        fallback_events: list[NoteEventAnnotation] = []
        for ne in note_events:
            nh_idx = ne.event_index if ne.event_index < len(noteheads) else -1
            has_stem = nh_idx in stem_map
            stem = stem_map.get(nh_idx)
            flags = flag_map.get(nh_idx, 0)
            if not ne.is_rest and nh_idx >= 0:
                num, den = classify_duration(
                    noteheads[nh_idx], has_stem, stem, flag_count=flags
                )
            else:
                num, den = 1, 4

            fallback_events.append(
                NoteEventAnnotation(
                    event_index=ne.event_index,
                    staff_index=ne.staff_index,
                    staff_step=ne.staff_step,
                    diatonic_step=ne.diatonic_step,
                    octave=ne.octave,
                    alter=ne.alter,
                    duration_num=num,
                    duration_den=den,
                    is_rest=ne.is_rest,
                    is_chord=ne.is_chord,
                    voice=ne.voice,
                    cx=ne.cx,
                    cy=ne.cy,
                    bbox=ne.bbox,
                )
            )

        ctx["note_events"] = fallback_events
        fallback_by_staff: dict[int, list[NoteEventAnnotation]] = {}
        for ne in fallback_events:
            fallback_by_staff.setdefault(ne.staff_index, []).append(ne)
        for staff_idx in fallback_by_staff:
            fallback_by_staff[staff_idx].sort(key=lambda e: e.cx)
        ctx["events_by_staff"] = fallback_by_staff
        ctx.setdefault("metrics", {})["duration_solver_rejected"] = 0
        return

    from notra.semantics.rhythm_solver import (
        build_candidates_from_events,
        decode_measure_rhythm,
    )

    per_measure = build_candidates_from_events(
        note_events, stem_map, flag_map, noteheads, measure_boundaries
    )

    # Debug: count candidates by flag_count
    flag_dist: dict[int, int] = {}
    for mc in per_measure:
        for c in mc:
            fc = c.flag_count
            flag_dist[fc] = flag_dist.get(fc, 0) + 1
    ctx.setdefault("metrics", {})["candidate_flag_dist"] = flag_dist

    # Decode each measure
    decoded_durations: dict[int, tuple[int, int]] = {}  # event_index → (ticks, unused)
    rejected_ids: set[str] = set()

    for measure_cands in per_measure:
        if not measure_cands:
            continue
        m_id = measure_cands[0].measure_id
        decode = decode_measure_rhythm(
            measure_cands,
            time_beats=time_beats,
            time_beat_type=time_beat_type,
            measure_id=m_id,
        )

        if not decode.valid:
            ctx.setdefault("warnings", []).append(
                f"measure {m_id}: rhythm invalid ({decode.diagnostics})"
            )

        for sel in decode.selected_events:
            try:
                if sel.candidate_id.startswith("evt_"):
                    evt_idx = int(sel.candidate_id.split("_")[1])
                    decoded_durations[evt_idx] = (sel.duration_ticks, 0)
            except (ValueError, IndexError):
                pass

        rejected_ids.update(decode.rejected_candidates)

    # Apply decoded durations and filter rejected candidates
    from fractions import Fraction
    filtered_events: list[NoteEventAnnotation] = []

    for idx, ne in enumerate(note_events):
        evt_key = ne.event_index
        cand_id = f"evt_{ne.event_index}"

        if cand_id in rejected_ids:
            continue  # skip false positives

        if evt_key in decoded_durations:
            ticks, _ = decoded_durations[evt_key]
            f = Fraction(ticks, 1920)  # ticks relative to whole note (1920)
            dur_num, dur_den = f.numerator, f.denominator
        else:
            nh_idx = ne.event_index if ne.event_index < len(noteheads) else -1
            has_stem = nh_idx in stem_map
            stem = stem_map.get(nh_idx)
            flags = flag_map.get(nh_idx, 0)
            if not ne.is_rest and nh_idx >= 0:
                num, den = classify_duration(
                    noteheads[nh_idx], has_stem, stem, flag_count=flags
                )
            else:
                num, den = 1, 4
            dur_num, dur_den = num, den

        filtered_events.append(
            NoteEventAnnotation(
                event_index=ne.event_index,
                staff_index=ne.staff_index,
                staff_step=ne.staff_step,
                diatonic_step=ne.diatonic_step,
                octave=ne.octave,
                alter=ne.alter,
                duration_num=dur_num,
                duration_den=dur_den,
                is_rest=ne.is_rest,
                is_chord=ne.is_chord,
                voice=ne.voice,
                cx=ne.cx,
                cy=ne.cy,
                bbox=ne.bbox,
            )
        )

    ctx["note_events"] = filtered_events
    ctx.setdefault("metrics", {})["duration_solver_rejected"] = len(rejected_ids)

    # Rebuild events_by_staff with filtered events
    filtered_by_staff: dict[int, list[NoteEventAnnotation]] = {}
    for ne in filtered_events:
        filtered_by_staff.setdefault(ne.staff_index, []).append(ne)
    for staff_idx in filtered_by_staff:
        filtered_by_staff[staff_idx].sort(key=lambda e: e.cx)
    ctx["events_by_staff"] = filtered_by_staff


# ---------------------------------------------------------------------------
# Stage 10: Voice assignment
# ---------------------------------------------------------------------------


def assign_voice_stage(ctx: dict[str, Any]) -> None:
    """Assign voice numbers based on stem direction."""
    noteheads: list[NoteheadCandidate] = ctx.get("notehead_candidates", [])
    note_events: list[NoteEventAnnotation] = ctx.get("note_events", [])
    stem_map: dict[int, Any] = ctx.get("stem_map", {})

    if not noteheads:
        return

    voice_map = assign_voice_from_stems(noteheads, stem_map)

    for idx, ne in enumerate(note_events):
        if ne.is_rest:
            continue
        nh_idx = ne.event_index
        voice = voice_map.get(nh_idx, 1)
        note_events[idx] = NoteEventAnnotation(
            event_index=ne.event_index,
            staff_index=ne.staff_index,
            staff_step=ne.staff_step,
            diatonic_step=ne.diatonic_step,
            octave=ne.octave,
            alter=ne.alter,
            duration_num=ne.duration_num,
            duration_den=ne.duration_den,
            is_rest=ne.is_rest,
            is_chord=ne.is_chord,
            voice=voice,
            cx=ne.cx,
            cy=ne.cy,
            bbox=ne.bbox,
        )

    ctx["note_events"] = note_events


# ---------------------------------------------------------------------------
# Stage 11: Measure assembly
# ---------------------------------------------------------------------------


def assemble_measures_stage(ctx: dict[str, Any]) -> None:
    """Build per-system measure boundaries from global barline positions.

    Each system gets its own copy of measure boundaries (same x-ranges,
    different system_index). Events are filtered by system in build_score_stage.
    """
    note_events: list[NoteEventAnnotation] = ctx.get("note_events", [])
    barline_xs: list[float] = ctx.get("barline_xs", [])
    barline_by_system: dict[int, list[float]] = ctx.get("barline_by_system", {})
    system_members: list[list[int]] = ctx.get("system_members", [])

    if not note_events:
        ctx["measure_boundaries"] = []
        ctx["events_by_staff"] = {}
        return

    # Group events by staff, sorted by x-position
    events_by_staff: dict[int, list[NoteEventAnnotation]] = {}
    for ne in note_events:
        events_by_staff.setdefault(ne.staff_index, []).append(ne)
    for staff_idx in events_by_staff:
        events_by_staff[staff_idx].sort(key=lambda e: e.cx)

    max_x = float(ctx.get("image_width", 2200))
    gray: np.ndarray | None = ctx.get("gray")
    bands: list[StaffBand] = ctx.get("staff_bands", [])

    # Build measure boundaries per system. Barlines can differ by system,
    # especially when detection includes staff-local stems/noise.
    all_measure_boundaries: list[MeasureBoundary] = []
    system_count = max(1, len(system_members))
    for sys_idx in range(system_count):
        sys_bars = sorted(barline_by_system.get(sys_idx, []))
        if not sys_bars:
            sys_bars = sorted(barline_xs)
        sys_left, sys_right = _system_x_extent(gray, bands, system_members, sys_idx, max_x)
        all_xs = [sys_left] + sys_bars
        interline_px = float(ctx.get("interline_px", 12.0))
        if not sys_bars or (sys_right - sys_bars[-1]) > max(10.0, interline_px * 2.0):
            all_xs.append(sys_right)
        for i in range(len(all_xs) - 1):
            x0, x1 = all_xs[i], all_xs[i + 1]
            if x1 - x0 < 10:
                continue
            all_measure_boundaries.append(
                MeasureBoundary(
                    measure_number=i + 1,
                    x_start=x0,
                    x_end=x1,
                    staff_index=0,
                    system_index=sys_idx,
                    barline_style="regular",
                )
            )

    ctx["measure_boundaries"] = all_measure_boundaries
    ctx["events_by_staff"] = events_by_staff


# ---------------------------------------------------------------------------
# Stage 12: Build Score IR
# ---------------------------------------------------------------------------


def build_score_stage(ctx: dict[str, Any]) -> None:
    """Assemble detected note events into a notra Score IR with multi-voice parts."""
    events_by_staff: dict[int, list[NoteEventAnnotation]] = ctx.get(
        "events_by_staff", {}
    )
    measure_boundaries: list[MeasureBoundary] = ctx.get("measure_boundaries", [])
    staff_anns: list[StaffAnnotation] = ctx.get("staff_annotations", [])
    bands: list[StaffBand] = ctx["staff_bands"]
    system_members: list[list[int]] = ctx.get("system_members", [])

    if not events_by_staff or not bands:
        ctx["errors"].append("No events or staff bands to build score from")
        return

    # Map staff → system index and position-within-system
    staff_system: dict[int, int] = {}
    staff_pos: dict[int, int] = {}
    for sys_idx, members in enumerate(system_members):
        for pos, si in enumerate(members):
            staff_system[si] = sys_idx
            staff_pos[si] = pos

    # --- Clef-pattern-based system refinement ---
    # For tightly-engraved scores (SATB, quartets), the gap-based system
    # detection may group multiple systems together. If a system has >6
    # bands and a repeating clef pattern is detected, split into the
    # detected group size.
    if system_members and staff_anns and len(bands) >= 6:
        clef_seq = [
            staff_anns[si].clef_sign if si < len(staff_anns) else "?"
            for si in range(len(bands))
        ]
        # Modulo-based pattern detection: clef_seq[i] == clef_seq[i+k]
        best_k = 1
        best_score = 0.0
        for k in (4, 3, 2):
            matches = 0
            total = 0
            for i in range(len(clef_seq) - k):
                total += 1
                if clef_seq[i] == clef_seq[i + k]:
                    matches += 1
            score = matches / max(total, 1)
            if score > best_score:
                best_score = score
                best_k = k

        if best_k > 1 and best_score > 0.4:
            new_members: list[list[int]] = []
            for i in range(0, len(bands), best_k):
                chunk = list(range(i, min(i + best_k, len(bands))))
                new_members.append(chunk)
            system_members = list(new_members)
            staff_system = {}
            staff_pos = {}
            for sys_idx, members in enumerate(system_members):
                for pos, si in enumerate(members):
                    staff_system[si] = sys_idx
                    staff_pos[si] = pos

    if not system_members:
        # Fallback: each staff is its own system
        for i in range(len(bands)):
            staff_system[i] = i
            staff_pos[i] = 0

    # Determine which staves are in the same system, then group by
    # position-within-system to form parts across systems.
    staff_part_map: dict[int, int] = {}
    for staff_idx in range(len(bands)):
        staff_part_map[staff_idx] = staff_pos.get(staff_idx, 0)

    # --- Grand-staff detection: pair staves only when clef pattern supports it ---
    # A grand-staff pair has a G-clef upper staff and F-clef lower staff.
    # SATB/quartets have repeating clef patterns (e.g. G,G,C,F) that should
    # NOT be paired into grand-staffs.
    grand_staff_pairs: set[tuple[int, int]] = set()
    for sys_idx, members in enumerate(system_members):
        if len(members) < 2:
            continue

        # Check if any consecutive pair has G+F clef (piano grand-staff)
        for i in range(len(members) - 1):
            s_upper = members[i]
            s_lower = members[i + 1]
            if s_upper >= len(staff_anns) or s_lower >= len(staff_anns):
                continue
            upper_clef = staff_anns[s_upper].clef_sign
            lower_clef = staff_anns[s_lower].clef_sign
            if upper_clef == "G" and lower_clef == "F":
                grand_staff_pairs.add((s_upper, s_lower))
                staff_part_map[s_upper] = staff_part_map[s_lower] = min(
                    staff_part_map[s_upper], staff_part_map[s_lower]
                )

    # Count unique positions to determine part count
    part_positions = sorted(set(staff_part_map.values()))
    part_count = len(part_positions)
    # Remap positions to contiguous 0,1,2...
    pos_remap = {old: new for new, old in enumerate(part_positions)}
    staff_part_map = {s: pos_remap[p] for s, p in staff_part_map.items()}

    # Build parts
    parts: list[Part] = []
    for part_idx in range(part_count):
        part_staves = sorted([s for s, p in staff_part_map.items() if p == part_idx])
        if not part_staves:
            continue

        # Determine part name from clef, position, and grand-staff status
        first_sa = staff_anns[part_staves[0]] if part_staves[0] < len(staff_anns) else None
        is_grand = len(part_staves) > 1 and any(
            (s0, s1) in grand_staff_pairs or (s1, s0) in grand_staff_pairs
            for s0 in part_staves for s1 in part_staves if s0 != s1
        )

        if is_grand:
            part_name = "Piano"
        elif len(part_staves) > 1:
            part_name = f"Part{part_idx + 1}"
        else:
            names = {0: "Soprano", 1: "Alto", 2: "Tenor", 3: "Bass"}
            part_name = names.get(part_idx, f"P{part_idx + 1}")
            if first_sa and first_sa.clef_sign == "F":
                part_name = "Bass"

        part_id = f"P{part_idx + 1}"

        # Determine which system this part belongs to
        part_sys = staff_system.get(part_staves[0], 0) if part_staves else 0

        # Build measures — only for boundaries in this part's system
        measures: list[Measure] = []
        for mb in measure_boundaries:
            # Filter: only use boundaries from this part's system
            if mb.system_index != part_sys:
                continue

            # Collect all events across all staves in this part for this measure
            all_voice_events: dict[int, list[Note | Rest]] = {}  # voice → events

            for s_idx in part_staves:
                events = events_by_staff.get(s_idx, [])
                measure_events = [e for e in events if mb.x_start <= e.cx < mb.x_end]
                measure_events.sort(key=lambda e: e.cx)

                for evt in measure_events:
                    v = evt.voice
                    all_voice_events.setdefault(v, [])

                    if evt.is_rest:
                        rest = Rest(
                            id=f"{part_id}_m{mb.measure_number}_s{s_idx}_r{len(all_voice_events[v])}",
                            duration=Duration(
                                numerator=evt.duration_num,
                                denominator=evt.duration_den,
                            ),
                            voice=v,
                        )
                        all_voice_events[v].append(rest)
                    else:
                        note = Note(
                            id=f"{part_id}_m{mb.measure_number}_s{s_idx}_n{len(all_voice_events[v])}",
                            pitch=Pitch(
                                step=evt.diatonic_step,
                                octave=evt.octave,
                                alter=evt.alter,
                            ),
                            duration=Duration(
                                numerator=evt.duration_num,
                                denominator=evt.duration_den,
                            ),
                            voice=v,
                        )
                        all_voice_events[v].append(note)

            if not all_voice_events:
                continue

            # Build Voice objects
            measure_voices: list[Voice] = []
            for v_num in sorted(all_voice_events.keys()):
                v_events = all_voice_events[v_num]
                # Events are already in x-order from insertion
                measure_voices.append(
                    Voice(
                        id=f"{part_id}_m{mb.measure_number}_v{v_num}",
                        events=v_events,
                    )
                )

            if not measure_voices:
                continue

            # --- Measure duration assertion ---
            # Verify total duration in each voice matches the time signature.
            # Fill gaps with rests so MusicXML is valid.
            time_beats = ctx.get("_structural_time_beats", 4)
            time_beat_type = ctx.get("_structural_time_beat_type", 4)
            from fractions import Fraction
            expected = Fraction(time_beats, time_beat_type)

            for voice in measure_voices:
                total = Fraction(0, 1)
                for evt in voice.events:
                    total += Fraction(evt.duration.numerator, evt.duration.denominator)

                gap = expected - total
                if gap > Fraction(0, 1):
                    # Insert a rest to fill the gap
                    gap_rest = Rest(
                        id=f"{voice.id}_fill",
                        duration=Duration(
                            numerator=gap.numerator,
                            denominator=gap.denominator,
                        ),
                        voice=int(voice.id.split('_v')[-1]) if '_v' in voice.id else 1,
                    )
                    voice.events.append(gap_rest)
                    ctx.setdefault("warnings", []).append(
                        f"{voice.id}: duration gap {float(gap):.3f} filled with rest"
                    )
                elif gap < Fraction(0, 1):
                    overflow = float(-gap)
                    expected_float = float(expected)
                    ctx.setdefault("warnings", []).append(
                        f"{voice.id}: duration overflow {overflow:.3f} "
                        f"(expected {expected_float})"
                    )

            # Compute divisions for this measure
            measure_denoms: set[int] = {1}
            for v_events in all_voice_events.values():
                for evt in v_events:
                    measure_denoms.add(evt.duration.denominator)
            measure_divs = 1
            for d in measure_denoms:
                from math import gcd
                measure_divs = measure_divs * d // gcd(measure_divs, d)
            measure_divs = max(measure_divs, 4)  # at least quarter-note resolution

            attrs = MeasureAttributes(
                clef=Clef(
                    sign=first_sa.clef_sign if first_sa else "G",
                    line=first_sa.clef_line if first_sa else 2,
                ),
                key=KeySignature(
                    fifths=ctx.get("_structural_key_fifths",
                                   first_sa.key_fifths if first_sa else 0),
                    mode="major",
                ),
                time=TimeSignature(
                    beats=ctx.get("_structural_time_beats", 4),
                    beat_type=ctx.get("_structural_time_beat_type", 4),
                ),
                divisions=measure_divs,
            )

            measure = Measure(
                id=f"{part_id}_m{mb.measure_number}",
                number=mb.measure_number,
                voices=measure_voices,
                attributes=attrs,
            )
            measures.append(measure)

        if measures:
            # Skip parts with zero notes (phantom staves from text/title fragments)
            total_notes = sum(
                1 for m in measures for v in m.voices for e in v.events
                if hasattr(e, 'pitch')
            )
            if total_notes == 0:
                ctx.setdefault("warnings", []).append(
                    f"Skipping empty part {part_name} (0 notes)"
                )
                continue
            part = Part(id=part_id, name=part_name, measures=measures)
            parts.append(part)

    if not parts:
        ctx["errors"].append("No parts could be assembled")
        return

    score = Score(
        id="notra_page_1",
        title="Recognized Score",
        parts=parts,
    )
    ctx["score"] = score


# ---------------------------------------------------------------------------
# Stage 13: Export MusicXML
# ---------------------------------------------------------------------------


def export_musicxml_stage(ctx: dict[str, Any]) -> None:
    """Export the Score IR to a MusicXML string."""
    score = ctx.get("score")
    if score is None:
        ctx["errors"].append("No score to export")
        return

    from notra.exporters.musicxml import export_score_to_musicxml
    musicxml = export_score_to_musicxml(score)
    ctx["musicxml"] = musicxml


# ---------------------------------------------------------------------------
# Structural classifier stage (CNN-based, replaces deterministic clef/key)
# ---------------------------------------------------------------------------


_structural_model_cache: Any = None


def classify_structure_stage(ctx: dict[str, Any]) -> None:
    """Run the structural CNN classifier to predict part count, clefs, key, time.

    Loads the trained checkpoint once and caches it across invocations.
    """
    global _structural_model_cache

    # Skip if predictions already injected (e.g. from batch runner)
    if ctx.get("_structural_part_count") is not None:
        return

    image_path = ctx.get("image_path")
    if not image_path:
        return

    try:
        if _structural_model_cache is None:
            # Find checkpoint relative to the project root
            import os
            from pathlib import Path

            from notra.pipeline.structural_classifier import load_structural_model
            repo_root = os.environ.get("NOTRA_ROOT", ".")
            ckpt = Path(repo_root) / "artifacts/training/structural_classifier/checkpoint.pt"
            if not ckpt.exists():
                ctx.setdefault("warnings", []).append(
                    "structural classifier checkpoint not found, using deterministic fallback"
                )
                return

            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _structural_model_cache = load_structural_model(ckpt)
            _structural_model_cache.to(dev)
            _structural_model_cache._device = dev

        from notra.pipeline.structural_classifier import (
            inject_structural_predictions,
            predict_structural,
        )
        preds = predict_structural(
            _structural_model_cache, image_path,
            device=getattr(_structural_model_cache, "_device", "cpu"),
        )
        inject_structural_predictions(ctx, preds)
    except Exception as exc:
        ctx.setdefault("warnings", []).append(
            f"structural classifier failed: {exc}, using deterministic fallback"
        )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


STAGE_ORDER: tuple[tuple[str, callable], ...] = (
    ("load_image", load_image_stage),
    ("detect_layout", detect_layout_stage),
    ("classify_structure", classify_structure_stage),
    ("detect_clefs", detect_clefs_stage),
    ("detect_noteheads", detect_noteheads_stage),
    ("detect_time_signature", detect_time_signature_stage),
    ("detect_rests", detect_rests_stage),
    ("detect_stems", detect_stems_stage),
    ("detect_accidentals", detect_accidentals_stage),
    ("assign_pitch", assign_pitch_stage),
    ("assemble_measures", assemble_measures_stage),
    ("assign_duration", assign_duration_stage),
    ("assign_voice", assign_voice_stage),
    ("build_score", build_score_stage),
    ("export_musicxml", export_musicxml_stage),
)


def run_full_pipeline(
    image_path: str,
    *,
    stages: tuple[tuple[str, callable], ...] | None = None,
    structural: dict[str, Any] | None = None,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the complete OMR recognition pipeline on an image.

    Args:
        image_path: Path to the input image (PNG, JPEG, etc.).
        stages: Optional custom stage list. Defaults to STAGE_ORDER.

    Returns:
        PipelineResult with annotations, score IR, MusicXML, and diagnostics.
    """
    if stages is None:
        stages = STAGE_ORDER

    ctx: dict[str, Any] = {
        "image_path": image_path,
        "errors": [],
        "warnings": [],
        "metrics": {},
    }
    if config is None:
        config = PipelineConfig.for_image(image_path)
    ctx.update(config.to_context())

    # Inject structural ground truth if provided (bypasses classifier)
    if structural:
        from notra.pipeline.structural_classifier import inject_structural_predictions
        inject_structural_predictions(ctx, structural)

    for stage_name, stage_fn in stages:
        try:
            stage_fn(ctx)
        except Exception as exc:
            ctx["errors"].append(f"[{stage_name}] {exc}")

    page_ann = PageAnnotations(
        image_width=ctx.get("image_width", 0),
        image_height=ctx.get("image_height", 0),
        staff_annotations=ctx.get("staff_annotations", []),
        note_events=ctx.get("note_events", []),
        measure_boundaries=ctx.get("measure_boundaries", []),
        barline_xs=ctx.get("barline_xs", []),
        interline_px=ctx.get("interline_px", 12.0),
    )

    return PipelineResult(
        page_annotations=page_ann,
        score=ctx.get("score"),
        musicxml=ctx.get("musicxml", ""),
        metrics=ctx.get("metrics", {}),
        errors=ctx["errors"],
        warnings=ctx["warnings"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_ink(ctx: dict[str, Any]) -> np.ndarray:
    """Get or compute binarized ink image.

    Uses Sauvola adaptive binarization — preserves anti-aliased edges
    that Otsu destroys, critical for stem/flag detection in renders.
    """
    if "ink" in ctx:
        return ctx["ink"]
    gray = ctx.get("gray")
    if gray is None:
        raise ValueError("No grayscale image in context")
    from notra.layout.staff import sauvola_binarize
    ink = sauvola_binarize(gray, window=15, k=0.15)
    ctx["ink"] = ink
    return ink


def _barlines_by_system_from_staff_barlines(
    staff_barlines: dict[int, list[float]],
    system_members: list[list[int]],
    bands: list[StaffBand],
) -> dict[int, list[float]]:
    """Merge staff-local barlines into per-system measure boundaries."""
    barline_by_system: dict[int, list[float]] = {}
    for sys_idx, members in enumerate(system_members):
        xs: list[float] = []
        for staff_idx in members:
            xs.extend(staff_barlines.get(staff_idx, []))
        interline = (
            float(np.median([bands[idx].interline_px for idx in members]))
            if members
            else 12.0
        )
        barline_by_system[sys_idx] = _dedupe_close_xs(
            xs,
            max_gap=max(4.0, interline * 0.60),
        )
    return barline_by_system


def _assign_global_barlines_to_systems(
    ink: np.ndarray,
    bands: list[StaffBand],
    system_members: list[list[int]],
    barline_xs: list[float],
) -> dict[int, list[float]]:
    """Assign legacy global vertical-run barlines to systems."""
    h_img, w_img = ink.shape
    barline_by_system: dict[int, list[float]] = {s: [] for s in range(len(system_members))}
    for x in barline_xs:
        xi = int(round(x))
        if xi < 0 or xi >= w_img:
            continue
        for sys_idx, members in enumerate(system_members):
            spanned = 0
            for si in members:
                band = bands[si]
                y0 = max(0, min(band.line_ys))
                y1 = min(h_img, max(band.line_ys) + 1)
                col = ink[y0:y1, xi]
                run = _longest_ink_run_stage(col)
                line_hits = 0
                for ly in band.line_ys:
                    line_y0 = max(0, int(ly) - 1)
                    line_y1 = min(h_img, int(ly) + 2)
                    if int(ink[line_y0:line_y1, xi].sum()) > 0:
                        line_hits += 1
                if run >= band.interline_px * 1.2 and line_hits >= 3:
                    spanned += 1
            if spanned >= max(1, len(members) * 0.4):
                barline_by_system[sys_idx].append(x)
    return barline_by_system


def _dedupe_close_xs(xs: list[float], *, max_gap: float) -> list[float]:
    if not xs:
        return []
    xs = sorted(xs)
    groups: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= max_gap:
            groups[-1].append(x)
            continue
        groups.append([x])
    return [float(np.median(group)) for group in groups]


def _system_x_extent(
    gray: np.ndarray | None,
    bands: list[StaffBand],
    system_members: list[list[int]],
    system_index: int,
    fallback_right: float,
) -> tuple[float, float]:
    if gray is None or not bands or system_index >= len(system_members):
        return 0.0, fallback_right

    extents: list[tuple[float, float]] = []
    for staff_idx in system_members[system_index]:
        if 0 <= staff_idx < len(bands):
            extents.append(estimate_staff_x_extent(gray, bands[staff_idx]))
    if not extents:
        return 0.0, fallback_right

    return min(left for left, _right in extents), max(right for _left, right in extents)


def _detect_system_members(bands: list[StaffBand]) -> list[list[int]]:
    """Group staff bands into systems based on vertical gaps.

    Uses adaptive thresholding: collects all inter-band gaps, clusters them
    into within-system and between-system populations, and splits accordingly.
    For scores with tight engraving (SATB, orchestral), the gap between
    systems may be only 10-20% larger than within-system staff gaps.

    Returns list of lists, each inner list is staff indices in one system.
    """
    if not bands:
        return []
    if len(bands) == 1:
        return [[0]]

    # Collect inter-band gaps
    gaps: list[float] = []
    for i in range(1, len(bands)):
        gap = float(bands[i].y_bottom - bands[i - 1].y_top)
        gaps.append(gap)

    # Use an interline-driven split threshold for small staff counts.
    # With only 2-4 detected bands (common in solo instrument pages),
    # the median gap can itself be an inter-system gap, which would block
    # all splitting if we use only median-based scaling.
    median_gap = float(np.median(gaps))
    interline = float(np.median([b.interline_px for b in bands]))
    if len(gaps) <= 3:
        threshold = interline * 6.0
    else:
        threshold = max(interline * 4.0, median_gap * 2.5)
        threshold = min(threshold, interline * 10.0)

    members: list[list[int]] = [[0]]
    for i in range(1, len(bands)):
        gap = gaps[i - 1]
        if gap > threshold:
            members.append([i])
        else:
            members[-1].append(i)

    # Post-process: detect repeating staff-group pattern and split accordingly.
    # For tightly-engraved scores (SATB, quartets), inter-system gaps may be
    # only 10-20% larger than within-system staff gaps — no simple threshold
    # works. Instead, detect the expected group size from gap periodicity.
    refined: list[list[int]] = []
    for group in members:
        if len(group) <= 8:
            refined.append(group)
            continue

        # Try to detect a repeating group size from the gap pattern.
        # Compute consecutive gaps within this group.
        group_gaps: list[float] = []
        for j in range(1, len(group)):
            g = float(bands[group[j]].y_bottom - bands[group[j - 1]].y_top)
            group_gaps.append(g)

        # The expected group size K is the dominant gap-count between
        # larger inter-system gaps.  Find the largest gap(s) and use the
        # count of smaller gaps between them as the group size.
        if len(group_gaps) < 2:
            refined.append(group)
            continue

        # Find system-break gaps: gaps > 1.6× the median gap.
        sys_break_indices: list[int] = []
        for j, g in enumerate(group_gaps):
            if g > median_gap * 1.6:
                sys_break_indices.append(j + 1)  # index in group where new system starts

        if sys_break_indices:
            # Split at each system break
            prev = 0
            for si in sys_break_indices:
                refined.append(group[prev:si])
                prev = si
            refined.append(group[prev:])
        else:
            # No clear breaks — use the largest internal gap
            best_split = -1
            best_gap = 0.0
            for j in range(1, len(group)):
                si_prev = group[j - 1]
                si_curr = group[j]
                g = float(bands[si_curr].y_bottom - bands[si_prev].y_top)
                if g > best_gap:
                    best_gap = g
                    best_split = j
            if best_split > 0 and best_gap > median_gap * 1.5:
                refined.append(group[:best_split])
                refined.append(group[best_split:])
            else:
                refined.append(group)

    return refined


def _longest_ink_run_stage(column: np.ndarray) -> int:
    """Find longest continuous ink run in a column slice."""
    max_run = 0
    run = 0
    for val in column.flat:
        if int(val) > 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run
