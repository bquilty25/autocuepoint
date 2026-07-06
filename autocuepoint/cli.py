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

from .analysis import bars_from_audio, bars_from_tempo, detect_phrase_boundaries, get_bar_duration
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
    "--no-backup",
    is_flag=True,
    default=False,
    help="Skip the automatic database backup. Not recommended.",
)
def main(
    offset: str,
    segments: int,
    overwrite: bool,
    track_filter: str | None,
    feature: str,
    min_duration: int,
    no_backup: bool,
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

    for track in tracks:
        total += 1
        label = f"{track.ArtistName or ''} - {track.Title or ''}".strip(" -") or str(track.ID)

        # Skip tracks that are too short
        track_duration = int(track.Length or 0)
        if track_duration < min_duration:
            skipped_short += 1
            continue

        # Skip if already has hot cues (unless --overwrite)
        if has_hot_cues(db, str(track.ID)) and not overwrite:
            click.echo(f"  [skip]    {label}  (already has hot cues; use --overwrite to replace)")
            skipped_existing += 1
            continue

        # Resolve audio path
        bpm_from_db, audio_path = get_track_bpm_and_path(track)

        if audio_path is None or not audio_path.exists():
            click.echo(f"  [missing] {label}  -- audio file not found, skipping", err=True)
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
            processed += 1

        except Exception as exc:  # noqa: BLE001
            click.echo(f"  [error]   {label}: {exc}", err=True)
            db.session.rollback()
            errored += 1

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
