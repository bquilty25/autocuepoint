"""
Cue point offset logic and hot cue colour assignment.

Takes raw phrase boundary times, applies a bar offset (moving each cue
point N bars earlier), and assigns hot cue slot numbers with colours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .xml_io import CuePoint

# ── Hot cue colour palette (8 slots, 0-indexed) ──────────────────────────────

# Each entry is (Red, Green, Blue)
HOT_CUE_COLOURS: list[tuple[int, int, int]] = [
    (255,   0,   0),   # 0 – Red
    (255, 128,   0),   # 1 – Orange
    (220, 200,   0),   # 2 – Yellow
    (  0, 200,   0),   # 3 – Green
    (  0, 200, 200),   # 4 – Cyan
    (  0, 100, 255),   # 5 – Blue
    (150,   0, 255),   # 6 – Purple
    (255,   0, 150),   # 7 – Pink
]


def build_hot_cues(
    boundary_times: np.ndarray,
    bar_duration: float,
    offset_bars: int,
    max_cues: int = 8,
    min_spacing_bars: int = 8,
    first_bar_time: float = 0.0,
) -> list[CuePoint]:
    """
    Convert phrase boundary times into a list of CuePoint objects.

    Uses furthest-point (maximin) selection to pick cues that are maximally
    spread across the track rather than just the first N in time order.

    Algorithm:
      1. Slot 0 is always `first_bar_time` (the first real downbeat from the
         ANLZ beat grid — not necessarily 0.0, since tracks may have silence
         or a count-in before the first musical bar).
      2. Apply the bar offset to all candidate boundaries; drop negatives.
      3. Iteratively pick the remaining candidate whose minimum distance to
         any already-selected cue is greatest, subject to min_spacing_bars.
      4. Sort accepted times chronologically and assign slots 0-7.

    Args:
        boundary_times: candidate boundary positions in seconds (may be more
                        than max_cues — the caller should over-supply)
        bar_duration: duration of one bar in seconds
        offset_bars: bars to subtract from each boundary (0, 16, or 32)
        max_cues: maximum hot cue slots to fill (cap at 8)
        min_spacing_bars: minimum bars between any two accepted cues
        first_bar_time: timestamp of the first musical downbeat in seconds
                        (from ANLZ); used as the fixed slot-0 anchor

    Returns:
        List of CuePoint objects, sorted chronologically, slots 0-7.
    """
    offset_seconds = offset_bars * bar_duration
    min_spacing_seconds = min_spacing_bars * bar_duration
    cap = min(max_cues, 8)

    # Build the candidate pool: apply offset, drop negatives, deduplicate
    candidates: list[float] = sorted({
        round(float(t) - offset_seconds, 3)
        for t in boundary_times
        if round(float(t) - offset_seconds, 3) > 0.0
    })

    # Slot 0 is the first real musical downbeat (not necessarily 0.0)
    selected: list[float] = [round(first_bar_time, 3)]

    # Furthest-point selection
    while len(selected) < cap and candidates:
        # For each candidate, compute its minimum distance to any selected cue
        def _min_dist(t: float) -> float:
            return min(abs(t - s) for s in selected)

        # Only consider candidates that satisfy the minimum spacing constraint
        valid = [t for t in candidates if _min_dist(t) >= min_spacing_seconds]
        if not valid:
            break

        # Pick the candidate furthest from all selected cues
        best = max(valid, key=_min_dist)
        selected.append(best)
        candidates.remove(best)

    # Sort chronologically and assign colour slots in time order
    selected.sort()
    cues: list[CuePoint] = []
    for slot, t in enumerate(selected):
        r, g, b = HOT_CUE_COLOURS[slot]
        cues.append(CuePoint(start=t, num=slot, name="", red=r, green=g, blue=b))

    return cues
