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
) -> list[CuePoint]:
    """
    Convert phrase boundary times into a list of CuePoint objects.

    Slot 0 is always placed at the very start of the track (0.0 s). The
    remaining slots are filled with the offset phrase boundaries in order,
    skipping any that also resolve to 0.0 s. Up to `max_cues` (max 8) cues
    are returned, assigned to hot cue slots 0-7.

    Args:
        boundary_times: array of phrase boundary positions in seconds
        bar_duration: duration of one bar in seconds
        offset_bars: number of bars to subtract from each boundary time
                     (0, 16, or 32 typically)
        max_cues: maximum number of cues to produce (cap at 8)

    Returns:
        List of CuePoint objects ready to pass to xml_io.write_cue_points().
    """
    offset_seconds = offset_bars * bar_duration
    cap = min(max_cues, 8)
    cues: list[CuePoint] = []

    # Slot 0 is always bar 1
    r, g, b = HOT_CUE_COLOURS[0]
    cues.append(CuePoint(start=0.0, num=0, name="", red=r, green=g, blue=b))

    slot = 1
    for boundary_time in boundary_times:
        if slot >= cap:
            break

        cue_time = round(float(boundary_time) - offset_seconds, 3)
        if cue_time <= 0.0:
            continue  # would duplicate the bar-1 cue

        cues.append(CuePoint(
            start=cue_time,
            num=slot,
            name="",
            red=r,
            green=g,
            blue=b,
        ))
        slot += 1

    return cues
