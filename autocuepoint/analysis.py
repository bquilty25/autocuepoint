"""
Audio analysis: bar position computation and phrase boundary detection.

Strategy:
1. If the track has TEMPO data from rekordbox, compute bar start times
   mathematically (most accurate for dance music with a locked grid).
2. If no TEMPO data, fall back to librosa beat tracking.
3. Extract bar-synchronous MFCC features and run librosa structural
   segmentation to find phrase boundaries.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np

from .xml_io import Tempo

# Suppress librosa/numba/audioread deprecation noise
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="audioread")

# ── Constants ────────────────────────────────────────────────────────────────

_HOP_LENGTH = 512
_SR = 22050  # librosa default; keeping low for speed
_MAX_AUDIO_BYTES = 500 * 1024 * 1024  # 500 MB hard limit before loading
_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".aiff", ".aif", ".flac",
    ".ogg", ".m4a", ".mp4", ".aac", ".wma",
}


# ── Bar position helpers ─────────────────────────────────────────────────────

def bars_from_tempo(tempos: list[Tempo], total_time: float) -> tuple[np.ndarray, float]:
    """
    Compute bar start times (in seconds) from rekordbox TEMPO data.

    Uses the first TEMPO element's anchor and BPM to project bar positions
    across the entire track. Multiple TEMPO elements (variable tempo) are
    handled by piecewise projection between anchors, sorted by Inizio.

    Returns:
        bar_times: 1-D array of bar start times in seconds (>= 0)
        bpm: representative BPM (from first TEMPO element)
    """
    if not tempos:
        raise ValueError("No TEMPO elements provided")

    sorted_tempos = sorted(tempos, key=lambda t: t.inizio)

    all_bar_times: list[float] = []

    for i, tempo in enumerate(sorted_tempos):
        bpm = tempo.bpm
        if bpm <= 0:
            continue

        beats_per_bar = _beats_per_bar(tempo.metro)
        bar_duration = (60.0 / bpm) * beats_per_bar

        # The anchor beat (Inizio) may not be on beat 1; project back to bar 1
        # Battito tells us which beat within the bar the anchor is
        beat_offset_in_bar = (tempo.battito - 1) * (60.0 / bpm)
        first_bar_start = tempo.inizio - beat_offset_in_bar

        # Project forward until the next tempo anchor (or end of track)
        end = sorted_tempos[i + 1].inizio if i + 1 < len(sorted_tempos) else total_time + bar_duration

        t = first_bar_start
        # Walk backwards to include any bars before the anchor
        while t - bar_duration >= -bar_duration * 0.5:
            t -= bar_duration
        # Now walk forward
        while t < end:
            if t >= 0:
                all_bar_times.append(t)
            t += bar_duration

    bar_times = np.unique(np.array(all_bar_times))
    bar_times = bar_times[bar_times <= total_time]
    valid_bpms = [t.bpm for t in sorted_tempos if t.bpm > 0]
    rep_bpm = valid_bpms[0] if valid_bpms else 0.0
    return bar_times, rep_bpm


def bars_from_audio(audio_path: Path) -> tuple[np.ndarray, float]:
    """
    Estimate bar start times from audio using librosa beat tracking.

    Assumes 4/4 time. Groups detected beats into groups of 4 and takes
    every 4th beat as a bar start.

    Returns:
        bar_times: 1-D array of bar start times in seconds
        bpm: estimated BPM
    """
    _validate_audio_path(audio_path)
    y, sr = librosa.load(str(audio_path), sr=_SR, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=_HOP_LENGTH)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=_HOP_LENGTH)

    bpm = float(np.atleast_1d(tempo)[0])

    # Group beats into bars of 4
    # Try to find the first downbeat via the onset strength envelope maximum
    # in the first few beats
    if len(beat_times) == 0:
        return np.array([]), bpm

    # Simple heuristic: start from beat 0, step by 4
    bar_indices = np.arange(0, len(beat_times), 4)
    bar_times = beat_times[bar_indices]
    return bar_times, bpm


# ── Phrase boundary detection ────────────────────────────────────────────────

def detect_phrase_boundaries(
    audio_path: Path,
    bar_times: np.ndarray,
    n_segments: int,
    feature: str = "mfcc",
) -> np.ndarray:
    """
    Detect phrase boundaries using librosa structural segmentation.

    Computes bar-synchronous features, builds a self-similarity matrix,
    and uses agglomerative clustering to find `n_segments` contiguous
    sections. Returns the boundary bar start times as a seconds array.

    Args:
        audio_path: path to the audio file
        bar_times: array of bar start times in seconds (from bars_from_tempo
                   or bars_from_audio)
        n_segments: number of segments to partition the track into
        feature: "mfcc" (default, better for electronic/dance music) or
                 "chroma" (better for harmonically rich pop/rock)

    Returns:
        boundary_times: 1-D array of boundary positions in seconds.
                        The first boundary is always the first bar (t=0 region).
    """
    if len(bar_times) == 0:
        return np.array([])

    _validate_audio_path(audio_path)
    y, sr = librosa.load(str(audio_path), sr=_SR, mono=True)

    # ── Extract frame-level features ─────────────────────────────────────────
    if feature == "chroma":
        frame_features = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_HOP_LENGTH)
    else:
        frame_features = librosa.feature.mfcc(y=y, sr=sr, hop_length=_HOP_LENGTH, n_mfcc=20)

    # ── Aggregate features per bar ────────────────────────────────────────────
    # Convert bar_times to frame indices in the feature matrix
    bar_frames = librosa.time_to_frames(bar_times, sr=sr, hop_length=_HOP_LENGTH)
    n_frames = frame_features.shape[1]
    bar_frames = np.clip(bar_frames, 0, n_frames - 1)

    n_bars = len(bar_times)
    bar_features = np.zeros((frame_features.shape[0], n_bars))

    for i in range(n_bars):
        start_f = bar_frames[i]
        end_f = bar_frames[i + 1] if i + 1 < n_bars else n_frames
        if end_f > start_f:
            bar_features[:, i] = frame_features[:, start_f:end_f].mean(axis=1)
        else:
            bar_features[:, i] = frame_features[:, start_f]

    # ── Structural segmentation ───────────────────────────────────────────────
    # Request 3x the desired cues so the selection layer has plenty of
    # candidates spread across the track to choose from.
    k = min(n_segments * 3, n_bars - 1)
    if k < 1:
        return bar_times[:1]

    # boundary_indices are bar indices where a new segment starts
    boundary_bar_indices = librosa.segment.agglomerative(bar_features, k=k)

    # Convert bar indices to seconds
    boundary_times = bar_times[boundary_bar_indices]
    return boundary_times


def get_bar_duration(bpm: float, metro: str = "4/4") -> float:
    """Return the duration of one bar in seconds."""
    if not bpm or bpm <= 0:
        raise ValueError(f"BPM must be positive, got {bpm!r}")
    beats_per_bar = _beats_per_bar(metro)
    return (60.0 / bpm) * beats_per_bar


def compute_raw_energy(audio_path: Path) -> float:
    """
    Compute a raw (unnormalised) energy value for a track.

    Combines RMS loudness (45 %), onset density (35 %), and spectral
    centroid/brightness (20 %) into a single float in [0.0, 1.0].
    Pass a list of these values to :func:`normalise_scores` to convert
    them to 1–5 star ratings relative to your library.

    Args:
        audio_path: path to a supported audio file

    Returns:
        Raw energy float in [0.0, 1.0].
    """
    _validate_audio_path(audio_path)
    y, sr = librosa.load(str(audio_path), sr=_SR, mono=True)

    # 1. Compute RMS and onset over the same set of frames, trimmed to the
    #    same length to guard against any off-by-one from different algorithms.
    rms_frames = librosa.feature.rms(y=y, hop_length=_HOP_LENGTH)[0]
    onset_env  = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP_LENGTH)
    n = min(len(rms_frames), len(onset_env))
    rms_frames = rms_frames[:n]
    onset_env  = onset_env[:n]

    # 2. RMS: 75th percentile of all non-trivial frames (> 1 % of max RMS)
    rms_threshold_low = float(rms_frames.max()) * 0.01
    active_mask = rms_frames > max(rms_threshold_low, 1e-4)
    active_rms = rms_frames[active_mask]
    rms = float(np.percentile(active_rms, 75)) if len(active_rms) > 0 else 0.0

    # 3. Onset: mean over BUSY frames only (above the 25th-percentile RMS of
    #    active frames) so quiet intros, breakdowns, and outros do not dilute
    #    the score for otherwise high-energy tracks.
    busy_threshold = float(np.percentile(active_rms, 25)) if len(active_rms) > 0 else 0.0
    busy_mask = active_mask & (rms_frames > busy_threshold)
    busy_onset = onset_env[busy_mask]
    onset_mean = float(np.mean(busy_onset)) if len(busy_onset) > 0 else float(np.mean(onset_env))

    # 4. Mean spectral centroid (brightness) — aligned to same frame count
    centroid = float(np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=_HOP_LENGTH)[0, :n]
    ))

    # Normalise each feature to [0, 1] using empirical bounds
    rms_norm = float(np.clip((rms - 0.01) / (0.45 - 0.01), 0.0, 1.0))
    onset_norm = float(np.clip((onset_mean - 0.05) / (2.0 - 0.05), 0.0, 1.0))
    centroid_norm = float(np.clip((centroid - 500) / (4000 - 500), 0.0, 1.0))

    # Weights: onset density (50%) > RMS loudness (30%) > brightness (20%)
    # Onset weighted highest so rhythmically intense tracks score high
    # regardless of mastering level.
    return float(0.30 * rms_norm + 0.50 * onset_norm + 0.20 * centroid_norm)


def normalise_scores(raw_values: list[float]) -> list[int]:
    """
    Convert a list of raw energy floats into 1–5 star ratings using
    percentile quintiles across the supplied batch.

    The bottom 20 % of tracks receive 1 star, the next 20 % receive 2, and
    so on, so every score tier is represented regardless of the absolute
    loudness of the library.

    Args:
        raw_values: list of raw energy floats (from :func:`compute_raw_energy`)

    Returns:
        List of integers in [1, 5], one per input value.
    """
    if not raw_values:
        return []
    arr = np.array(raw_values)
    thresholds = [float(np.percentile(arr, p)) for p in (20, 40, 60, 80)]
    scores = []
    for v in raw_values:
        if v <= thresholds[0]:
            scores.append(1)
        elif v <= thresholds[1]:
            scores.append(2)
        elif v <= thresholds[2]:
            scores.append(3)
        elif v <= thresholds[3]:
            scores.append(4)
        else:
            scores.append(5)
    return scores


# ── Internal helpers ─────────────────────────────────────────────────────────

def _validate_audio_path(audio_path: Path) -> None:
    """Raise ValueError if the path has an unsupported extension or is too large."""
    if audio_path.suffix.lower() not in _AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio file type {audio_path.suffix!r}. "
            f"Supported: {', '.join(sorted(_AUDIO_EXTENSIONS))}"
        )
    size = audio_path.stat().st_size
    if size > _MAX_AUDIO_BYTES:
        raise ValueError(
            f"Audio file is {size / (1024**2):.0f} MB, exceeding the 500 MB limit: "
            f"{audio_path.name}"
        )

def _beats_per_bar(metro: str) -> int:
    """Parse a time signature string like '4/4' and return the numerator."""
    if not metro:
        return 4
    try:
        return int(metro.split("/")[0])
    except (IndexError, ValueError):
        return 4
