"""
Structural classifier integration for the OMR pipeline.

Loads the trained checkpoint and injects structural predictions
(part count, clefs, key, time) into the pipeline context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_structural_model(checkpoint_path: str | Path) -> Any:
    """Load the trained structural classifier checkpoint."""
    import torch
    from scripts.train_structural_classifier import StructuralCNN

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = StructuralCNN(image_size=checkpoint.get("image_size", 224))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_structural(model: Any, image_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Run the structural classifier on a page image.

    Returns dict with:
        part_count, clef_signs (per-part), key_fifths, time_beats, time_beat_type
    """
    import torch

    with Image.open(image_path) as img:
        rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device)

    with torch.no_grad():
        part_logits, key_logits, beats_logits, bt_logits, clef_logits = model(tensor)

    part_count = part_logits.argmax(1).item()
    key_class = key_logits.argmax(1).item()
    key_fifths = key_class - 7
    time_beats = beats_logits.argmax(1).item()
    bt_class = bt_logits.argmax(1).item()
    bt_map = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32}
    time_beat_type = bt_map.get(bt_class, 4)

    clef_preds = clef_logits.argmax(-1).squeeze(0).tolist()
    clef_map = {0: "G", 1: "F", 2: "C", 3: None}
    clef_signs: list[tuple[str, int]] = []
    for c in clef_preds[:part_count]:
        sign = clef_map.get(c)
        if sign == "F":
            clef_signs.append(("F", 4))
        elif sign == "C":
            clef_signs.append(("C", 3))
        else:
            clef_signs.append(("G", 2))

    return {
        "part_count": part_count,
        "clef_signs": clef_signs,
        "key_fifths": key_fifths,
        "time_beats": time_beats,
        "time_beat_type": time_beat_type,
    }


def inject_structural_predictions(ctx: dict[str, Any], predictions: dict[str, Any]) -> None:
    """Inject classifier predictions into the pipeline context.

    This overrides the deterministic clef/key/time detections with
    classifier outputs.
    """
    ctx["_structural_part_count"] = predictions["part_count"]
    ctx["_structural_clefs"] = predictions["clef_signs"]
    ctx["_structural_key_fifths"] = predictions["key_fifths"]
    ctx["_structural_time_beats"] = predictions["time_beats"]
    ctx["_structural_time_beat_type"] = predictions["time_beat_type"]
