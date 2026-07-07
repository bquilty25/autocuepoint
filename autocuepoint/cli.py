"""
CLI entry point for autocuepoint.

Writes hot cue points directly to the rekordbox 6+ database (master.db),
which is the only reliable method for rekordbox 6/7 since the XML import
menu option was removed in rekordbox 6.

Usage:
    autocuepoint --offset 16
    autocuepoint --offset 32 --segments 6 --overwrite
    autocuepoint --offset 0 --track "Daft Punk"
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

from .analysis import bars_from_audio, bars_from_tempo, absolute_scores, compute_raw_energy, detect_phrase_boundaries, get_bar_duration, normalise_scores
from .cues import build_hot_cues
from .db_io import (
    backup_database,
    delete_hot_cues,
    get_all_tracks,
    get_bar_times_from_anlz,
    get_track_bpm_and_path,
    get_tracks_by_name,
    has_hot_cues,
    open_db,
    write_cues_to_db,
    write_energy_score,
)
from .xml_io import Tempo as XmlTempo


@click.command()
@click.option(
    "--offset",
    default=0,
    type=click.Choice(["0", "16", "32"]),
    show_default=True,
    help="Shift each cue point this many bars before the detected phrase boundary.",
)
@click.option(
    "--segments",
    default=8,
    show_default=True,
    type=click.IntRange(1, 8),
    help="Number of phrase segments (and therefore cue points) to detect per track.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="If set, replace existing hot cues. Without this flag, tracks that already "
         "have hot cues are skipped.",
)
@click.option(
    "--track",
    "track_filter",
    default=None,
    type=str,
    help="Case-insensitive substring filter: only process tracks whose artist or "
         "name contains this string.",
)
@click.option(
    "--feature",
    default="mfcc",
    type=click.Choice(["mfcc", "chroma"]),
    show_default=True,
    help="Audio feature used for structural segmentation. "
         "'mfcc' works best for electronic/dance music; "
         "'chroma' works best for harmonically rich pop/rock.",
)
@click.option(
    "--min-duration",
    default=60,
    show_default=True,
    type=int,
    help="Skip tracks shorter than this many seconds (e.g. sample loops).",
)
@click.option(
    "--min-spacing",
    default=8,
    show_default=True,
    type=click.IntRange(1, 64),
    help="Minimum number of bars between consecutive cue points. "
         "Cues closer than this are dropped.",
)
@click.option(
    "--energy-score",
    is_flag=True,
    default=False,
    help="Compute a perceived energy score (1–5) for each track and write it to "
         "the rekordbox star rating field. Based on onset density and spectral "
         "brightness. WARNING: overwrites any existing star ratings.",
)
@click.option(
    "--normalise/--no-normalise",
    "normalise",
    default=True,
    show_default=True,
    help="With --normalise (default): scores are percentile quintiles relative "
         "to your library — every tier gets 20%% of tracks. "
         "With --no-normalise: fixed absolute thresholds are used — scores "
         "reflect the track's audio characteristics regardless of the rest of "
         "the library.",
)
@click.option(
    "--no-backup",
    is_flag=True,
    default=False,
    help="Skip the automatic database backup. Not recommended.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print extra detail: short-duration skips, missing tracks, and cue timestamps.",
)
def main(
    offset: str,
    segments: int,
    overwrite: bool,
    track_filter: str | None,
    feature: str,
    min_duration: int,
    min_spacing: int,
    energy_score: bool,
    normalise: bool,
    no_backup: bool,
    verbose: bool,
) -> None:
    """Automatically add hot cue points to the rekordbox 6/7 library database."""

    offset_bars = int(offset)

    # Safety: backup the database first
    if not no_backup:
        try:
            backup_path = backup_database()
            click.echo(f"Backup created: {backup_path.name}")
        except Exception as exc:
            click.echo(f"WARNING: Could not create backup: {exc}", err=True)
            if not click.confirm("Continue without backup?", default=False):
                sys.exit(1)

    # Open database
    click.echo("Opening rekordbox database ...")
    try:
        db = open_db()
    except Exception as exc:
        click.echo(f"ERROR: Could not open rekordbox database: {exc}", err=True)
        click.echo(
            "Make sure rekordbox is installed and has been opened at least once.",
            err=True,
        )
        sys.exit(1)

    # Get tracks
    if track_filter:
        tracks = get_tracks_by_name(db, track_filter)
    else:
        tracks = get_all_tracks(db)

    total = 0
    skipped_existing = 0
    skipped_no_audio = 0
    skipped_short = 0
    processed = 0
    errored = 0
    # Accumulate (track_obj, raw_energy) pairs for batch normalisation
    energy_pending: list[tuple[object, float]] = []

    for track in tracks:
        total += 1
        label = f"{track.ArtistName or ''} - {track.Title or ''}".strip(" -") or str(track.ID)

        # Skip tracks that are too short
        track_duration = int(track.Length or 0)
        if track_duration < min_duration:
            if verbose:
                click.echo(f"  [short]   {label}  ({track_duration}s < {min_duration}s min)")
            skipped_short += 1
            continue

        # Skip if already has hot cues (unless --overwrite)
        if has_hot_cues(db, str(track.ID)) and not overwrite:
            if verbose:
                click.echo(f"  [skip]    {label}  (already has hot cues; use --overwrite to replace)")
            skipped_existing += 1
            continue

        # Resolve audio path
        bpm_from_db, audio_path = get_track_bpm_and_path(track)

        if audio_path is None or not audio_path.exists():
            if verbose:
                click.echo(f"  [missing] {label}  -- audio file not found, skipping")
            skipped_no_audio += 1
            continue

        click.echo(f"  [analyse] {label}")

        try:
            # Compute bar positions:
            # 1st choice: exact ANLZ beat grid (on-grid, matches rekordbox waveform)
            # 2nd choice: synthesise from rekordbox AverageBpm
            # 3rd choice: librosa beat detection
            bar_times, bpm = get_bar_times_from_anlz(track)
            if bar_times is not None and len(bar_times) > 0:
                source = f"ANLZ beat grid ({bpm:.1f} BPM)" if bpm else "ANLZ beat grid"
            else:
                bpm = bpm_from_db or 0.0
                if bpm > 0:
                    synth_tempo = XmlTempo(inizio=0.0, bpm=bpm, metro="4/4", battito=1)
                    bar_times, bpm = bars_from_tempo([synth_tempo], float(track.Length or 600))
                    source = f"rekordbox BPM ({bpm:.1f})"
                else:
                    bar_times, bpm = bars_from_audio(audio_path)
                    source = f"librosa analysis ({bpm:.1f} BPM)"

            if len(bar_times) == 0:
                click.echo(f"           -> No bars detected, skipping", err=True)
                skipped_no_audio += 1
                continue

            click.echo(f"           -> {len(bar_times)} bars detected via {source}")

            if not bpm or bpm <= 0:
                click.echo(f"           -> BPM unavailable, skipping", err=True)
                errored += 1
                continue

            # Detect phrase boundaries
            boundary_times = detect_phrase_boundaries(
                audio_path=audio_path,
                bar_times=bar_times,
                n_segments=segments,
                feature=feature,
            )

            if len(boundary_times) == 0:
                click.echo(f"           -> No boundaries detected, skipping", err=True)
                errored += 1
                continue

            # Build cue points with offset
            bar_duration = get_bar_duration(bpm)
            cues = build_hot_cues(
                boundary_times=boundary_times,
                bar_duration=bar_duration,
                offset_bars=offset_bars,
                max_cues=segments,
                min_spacing_bars=min_spacing,
                first_bar_time=float(bar_times[0]) if len(bar_times) > 0 else 0.0,
            )

            if not cues:
                click.echo(
                    f"           -> All cue positions fell before 0 s with "
                    f"--offset {offset_bars}; try a smaller offset."
                )
                errored += 1
                continue

            # Write to database
            if overwrite:
                deleted = delete_hot_cues(db, str(track.ID))
                if deleted:
                    click.echo(f"           -> Removed {deleted} existing hot cues")

            n_written = write_cues_to_db(db, track, cues)
            click.echo(f"           -> {n_written} hot cues written (offset -{offset_bars} bars)")
            if verbose:
                for cue in cues:
                    mins, secs = divmod(cue.start, 60)
                    click.echo(f"              cue {cue.num + 1}: {int(mins):02d}:{secs:05.2f}  {cue.name or ''}")

            # Optional energy score — collect raw value for batch normalisation
            if energy_score:
                raw = compute_raw_energy(audio_path)
                energy_pending.append((track, raw))

            processed += 1

        except Exception as exc:  # noqa: BLE001
            click.echo(f"  [error]   {label}: {exc}", err=True)
            db.session.rollback()
            errored += 1

    # Normalise and write energy scores across the whole batch
    if energy_pending:
        raw_values = [r for _, r in energy_pending]
        if normalise:
            scores = normalise_scores(raw_values)
            mode_label = "normalised (percentile quintiles)"
        else:
            scores = absolute_scores(raw_values)
            mode_label = "absolute (fixed thresholds)"
        click.echo(f"\nWriting energy scores for {len(scores)} tracks ({mode_label}) ...")
        from collections import Counter
        dist = Counter(scores)
        for s in sorted(dist):
            click.echo(f"  {'★' * s}{'☆' * (5 - s)}  {dist[s]:4d}")
        for (track_obj, _), score in zip(energy_pending, scores):
            write_energy_score(db, track_obj, score)

    # Commit
    if processed > 0:
        click.echo("\nCommitting to database ...")
        db.session.commit()
        click.echo("Done. Restart rekordbox to see the new cue points.")
    else:
        click.echo("\nNo tracks were modified; database not changed.")

    # Summary
    click.echo(
        f"\nSummary: {total} tracks scanned | "
        f"{processed} processed | "
        f"{skipped_existing} skipped (existing cues) | "
        f"{skipped_no_audio} skipped (audio not found) | "
        f"{skipped_short} skipped (too short) | "
        f"{errored} errors"
    )


if __name__ == "__main__":
    main()


# ── Restore command ───────────────────────────────────────────────────────────

_DB_PATH = Path.home() / "Library" / "Pioneer" / "rekordbox" / "master.db"


@click.command()
@click.option(
    "--file",
    "backup_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a specific backup file to restore. If omitted, an interactive "
         "list of available autocuepoint backups is shown.",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
def restore(backup_file: Path | None, yes: bool) -> None:
    """Restore master.db from an autocuepoint backup."""
    import shutil
    import subprocess
    from datetime import datetime

    # Abort if rekordbox is running
    result = subprocess.run(["pgrep", "-x", "rekordbox"], capture_output=True)
    if result.returncode == 0:
        click.echo(
            "ERROR: rekordbox is currently running. Close it before restoring.",
            err=True,
        )
        sys.exit(1)

    # Resolve which backup to restore
    if backup_file is None:
        backups = sorted(_DB_PATH.parent.glob("master.backup_autocue_*.db"))
        if not backups:
            click.echo("No autocuepoint backups found in "
                       f"{_DB_PATH.parent}", err=True)
            sys.exit(1)

        click.echo("Available backups (most recent last):\n")
        for i, p in enumerate(backups, 1):
            # Parse timestamp from filename for display
            try:
                ts = p.stem.split("master.backup_autocue_")[1]
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                label = dt.strftime("%Y-%m-%d  %H:%M:%S")
            except (IndexError, ValueError):
                label = p.stem
            size_mb = p.stat().st_size / (1024 ** 2)
            click.echo(f"  [{i}]  {label}  ({size_mb:.1f} MB)  {p.name}")

        click.echo()
        choice = click.prompt(
            "Enter number to restore (or q to quit)",
            default="q",
        )
        if choice.strip().lower() == "q":
            click.echo("Cancelled.")
            sys.exit(0)
        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(backups)):
                raise ValueError
        except ValueError:
            click.echo("Invalid selection.", err=True)
            sys.exit(1)
        backup_file = backups[idx]

    # Confirm
    click.echo(f"\nRestore:  {backup_file.name}")
    click.echo(f"      ->  {_DB_PATH}")
    if not yes:
        click.confirm("This will overwrite your current library. Continue?",
                      abort=True)

    # Safety: snapshot the current database before overwriting
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pre_restore_backup = _DB_PATH.with_name(f"master.pre_restore_{ts}.db")
    shutil.copy2(_DB_PATH, pre_restore_backup)
    click.echo(f"\nSafety backup of current state: {pre_restore_backup.name}")

    # Perform the restore
    shutil.copy2(backup_file, _DB_PATH)

    # Integrity check
    if _DB_PATH.stat().st_size != backup_file.stat().st_size:
        # Roll back to the safety snapshot
        shutil.copy2(pre_restore_backup, _DB_PATH)
        click.echo(
            "ERROR: Restored file size does not match backup. "
            "Original database has been preserved.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Restored successfully. Open rekordbox to verify.")

