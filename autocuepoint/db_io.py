"""
Direct rekordbox database I/O via pyrekordbox.

Reads track info from the master.db and writes hot cue points into the
djmdCue table. This avoids the XML import/export round-trip that rekordbox
6+ no longer supports via a simple menu option.

WARNING: Always close rekordbox before running this tool, or at minimum
do not save/sync in rekordbox while the tool is running, to avoid the
database being overwritten by rekordbox on exit.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .cues import CuePoint


def _num_to_kind(num: int) -> int:
    """
    Convert a hot cue slot number (0-7, XML Num attribute) to a rekordbox
    database Kind value.

    rekordbox stores hot cues as Kind 1-9, skipping Kind=4 which is reserved
    for Loop markers.

        Num:  0  1  2  3  4  5  6  7
        Kind: 1  2  3  5  6  7  8  9
    """
    if num <= 2:
        return num + 1
    return num + 2  # skip Kind=4 (Loop)


def _seconds_to_msec(seconds: float) -> int:
    return round(seconds * 1000)


def _msec_to_frame(msec: int) -> int:
    """rekordbox uses an internal frame rate of 150 fps."""
    return round(msec * 150 / 1000)


def backup_database() -> Path:
    """
    Create a timestamped backup of master.db before writing.

    Verifies the copy completed (size check) and prunes old autocue
    backups, keeping the five most recent.

    Returns the path of the backup file.
    """
    db_path = Path.home() / "Library" / "Pioneer" / "rekordbox" / "master.db"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"master.backup_autocue_{ts}.db")
    shutil.copy2(db_path, backup_path)

    # Integrity check: backup must be the same size as the source
    if backup_path.stat().st_size != db_path.stat().st_size:
        backup_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Backup copy is incomplete (size mismatch). Aborting to protect the database."
        )

    # Prune old autocue backups, keeping the 5 most recent
    old_backups = sorted(db_path.parent.glob("master.backup_autocue_*.db"))
    for stale in old_backups[:-5]:
        stale.unlink(missing_ok=True)

    return backup_path


def open_db():
    """Open the rekordbox 6 database and return a Rekordbox6Database instance.

    Raises RuntimeError if rekordbox is currently running, because rekordbox
    overwrites master.db on exit and would silently discard any cues written
    by this tool.
    """
    import subprocess
    import pyrekordbox

    result = subprocess.run(["pgrep", "-x", "rekordbox"], capture_output=True)
    if result.returncode == 0:
        raise RuntimeError(
            "rekordbox is currently running. Close it before running autocuepoint "
            "to prevent rekordbox from overwriting the database on exit."
        )

    return pyrekordbox.Rekordbox6Database()


def get_all_tracks(db) -> list:
    """Return all djmdContent rows that have a BPM > 0."""
    return [t for t in db.get_content() if (t.BPM or 0) > 0]


def get_tracks_by_name(db, name_filter: str) -> list:
    """Return djmdContent rows whose title or artist matches a substring."""
    lf = name_filter.lower()
    return [
        t for t in db.get_content()
        if (t.BPM or 0) > 0
        and (lf in (t.Title or "").lower() or lf in (t.ArtistName or "").lower())
    ]


def has_hot_cues(db, content_id: str) -> bool:
    """Return True if the track already has any hot cue entries (Kind > 0)."""
    cues = list(db.get_cue(ContentID=content_id))
    return any(c.Kind is not None and c.Kind > 0 for c in cues)


def delete_hot_cues(db, content_id: str) -> int:
    """Delete all hot cue rows (Kind > 0) for a track. Returns count deleted."""
    cues = [c for c in db.get_cue(ContentID=content_id)
            if c.Kind is not None and c.Kind > 0]
    for c in cues:
        db.session.delete(c)
    return len(cues)


def write_cues_to_db(db, content: object, cues: list[CuePoint]) -> int:
    """
    Insert hot cue rows into djmdCue for the given djmdContent row.

    Each CuePoint is inserted as a single hot cue entry (no paired memory cue
    is needed when writing directly to the database).

    Returns the number of cues written.
    """
    from pyrekordbox.db6 import tables

    now = datetime.now(timezone.utc).replace(tzinfo=None)  # rekordbox stores naive UTC

    for cue in cues:
        in_msec = _seconds_to_msec(cue.start)
        new_id = str(db.generate_unused_id(tables.DjmdCue))

        row = tables.DjmdCue(
            ID=new_id,
            ContentID=str(content.ID),
            ContentUUID=content.UUID,
            UUID=str(uuid4()),
            InMsec=in_msec,
            InFrame=_msec_to_frame(in_msec),
            InMpegFrame=0,
            InMpegAbs=0,
            OutMsec=None,
            OutFrame=None,
            OutMpegFrame=None,
            OutMpegAbs=None,
            Kind=_num_to_kind(cue.num),
            Color=-1,
            ColorTableIndex=None,
            ActiveLoop=0,
            Comment=cue.name or None,
            BeatLoopSize=0,
            CueMicrosec=None,
            InPointSeekInfo=None,
            OutPointSeekInfo=None,
            rb_data_status=0,
            rb_local_data_status=0,
            rb_local_deleted=0,
            rb_local_synced=0,
            usn=None,
            rb_local_usn=None,
            created_at=now,
            updated_at=now,
        )
        db.add(row)

    db.session.flush()
    return len(cues)


def get_track_bpm_and_path(track) -> tuple[float | None, Path | None]:
    """Extract BPM and audio file path from a djmdContent row."""
    # BPM is stored as centiBPM (x100) in the database
    raw_bpm = track.BPM
    bpm: float | None = None
    if raw_bpm:
        candidate = float(raw_bpm) / 100.0
        if candidate > 0:
            bpm = candidate

    path: Path | None = None
    folder_path = track.FolderPath or ""
    if folder_path:
        # Reject Spotify URIs and other non-filesystem schemes
        if folder_path.startswith("spotify:") or "://" in folder_path:
            return bpm, None
        p = Path(folder_path)
        # Reject paths containing traversal components
        if ".." in p.parts:
            return bpm, None
        path = p

    return bpm, path


_RB_SHARE = Path.home() / "Library/Pioneer/rekordbox/share"


def get_bar_times_from_anlz(track) -> tuple[np.ndarray | None, float | None]:
    """
    Read exact bar start times from the rekordbox ANLZ beat grid file.

    Returns (bar_times_seconds, bpm) if the ANLZ file is available and
    parseable, otherwise (None, None).

    bar_times_seconds is a numpy array of downbeat timestamps in seconds,
    directly from rekordbox's analysed beat grid (so they are always on-grid).
    """
    import numpy as np

    anlz_path_rel = track.AnalysisDataPath
    if not anlz_path_rel:
        return None, None

    anlz_path = _RB_SHARE / anlz_path_rel.lstrip("/")

    # Reject paths that escape the expected share directory (path traversal guard)
    try:
        anlz_path.resolve().relative_to(_RB_SHARE.resolve())
    except ValueError:
        return None, None

    if not anlz_path.exists():
        return None, None

    try:
        import pyrekordbox
        anlz = pyrekordbox.AnlzFile.parse_file(str(anlz_path))
        tag = anlz.get_tag("PQTZ")
        if tag is None:
            return None, None

        # get_times() returns beat timestamps in seconds
        beat_times = tag.get_times()
        beat_nums = tag.beats  # 1=downbeat, 2, 3, 4

        bar_mask = np.asarray(beat_nums) == 1
        bar_times = np.asarray(beat_times)[bar_mask]

        # Derive BPM from inter-beat intervals (more accurate than DB value)
        all_times = np.asarray(beat_times)
        db_bpm = float(track.BPM) / 100.0 if track.BPM else None
        if len(all_times) > 1:
            ibi = np.median(np.diff(all_times))  # inter-beat interval in seconds
            bpm = 60.0 / ibi if ibi > 0 else db_bpm
        else:
            bpm = db_bpm

        return bar_times, bpm
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug(
            "ANLZ read failed for %s: %s", anlz_path, exc, exc_info=True
        )
        return None, None
