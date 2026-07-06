"""
Rekordbox XML I/O helpers.

Reads and writes the rekordbox DJ_PLAYLISTS XML format, handling:
- Decoding track Location URLs to filesystem paths
- Extracting TEMPO beat grid information
- Reading and writing POSITION_MARK (cue point) elements
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import defusedxml.ElementTree as ET
import xml.etree.ElementTree as _stdlib_ET  # used only for building/writing


@dataclass
class Tempo:
    """Represents a single TEMPO beat grid anchor."""
    inizio: float       # anchor beat position in seconds
    bpm: float          # BPM at this anchor
    metro: str          # time signature, e.g. "4/4"
    battito: int        # beat number within bar (1 = downbeat)


@dataclass
class CuePoint:
    """A cue point to write into a POSITION_MARK element."""
    start: float        # position in seconds
    num: int            # -1 = memory cue, 0-7 = hot cue slot
    name: str = ""
    red: int | None = None
    green: int | None = None
    blue: int | None = None


@dataclass
class TrackInfo:
    """Parsed information for a single TRACK element."""
    element: ET.Element
    track_id: str
    name: str
    artist: str
    location_url: str           # raw Location attribute value
    audio_path: Path | None     # decoded filesystem path, or None if unresolvable
    average_bpm: float | None
    tempos: list[Tempo] = field(default_factory=list)
    existing_hot_cue_nums: set[int] = field(default_factory=set)


def _decode_location(location_url: str) -> Path | None:
    """
    Convert a rekordbox file://localhost/... URL to a filesystem Path.

    Rekordbox uses percent-encoded file URLs on both macOS and Windows.
    On Windows the path starts with a drive letter after the third slash.
    """
    if not location_url.startswith("file://"):
        return None
    # Strip the scheme and optional hostname
    # file://localhost/path -> /path (macOS)
    # file://localhost/D:/path -> D:/path (Windows)
    without_scheme = location_url[len("file://"):]
    # Remove the hostname segment (everything up to the next /)
    if without_scheme.startswith("localhost"):
        without_scheme = without_scheme[len("localhost"):]
    # without_scheme now starts with /
    decoded = urllib.parse.unquote(without_scheme)
    # On Windows, strip the leading slash before the drive letter
    if len(decoded) >= 3 and decoded[0] == "/" and decoded[2] == ":":
        decoded = decoded[1:]
    return Path(decoded)


def _encode_location(path: Path) -> str:
    """Convert a filesystem Path back to a rekordbox file://localhost/... URL."""
    posix = path.as_posix()
    # On Windows paths start with a drive letter; we need to add a leading /
    if len(posix) >= 2 and posix[1] == ":":
        posix = "/" + posix
    encoded = urllib.parse.quote(posix, safe="/:")
    return f"file://localhost{encoded}"


def parse_xml(xml_path: Path) -> tuple[ET.ElementTree, list[TrackInfo]]:
    """
    Parse a rekordbox XML export.

    Returns the ElementTree (for later writing) and a list of TrackInfo objects,
    one per TRACK element found in the COLLECTION.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    collection = root.find("COLLECTION")
    if collection is None:
        return tree, []

    tracks: list[TrackInfo] = []
    for elem in collection.findall("TRACK"):
        location_url = elem.get("Location", "")
        audio_path = _decode_location(location_url) if location_url else None

        bpm_str = elem.get("AverageBpm")
        average_bpm = float(bpm_str) if bpm_str else None

        tempos: list[Tempo] = []
        for t in elem.findall("TEMPO"):
            try:
                tempos.append(Tempo(
                    inizio=float(t.get("Inizio", 0)),
                    bpm=float(t.get("Bpm", 0)),
                    metro=t.get("Metro", "4/4"),
                    battito=int(t.get("Battito", 1)),
                ))
            except (ValueError, TypeError):
                pass

        existing_hot_cue_nums: set[int] = set()
        for pm in elem.findall("POSITION_MARK"):
            num_str = pm.get("Num", "-1")
            try:
                num = int(num_str)
                if 0 <= num <= 7:
                    existing_hot_cue_nums.add(num)
            except ValueError:
                pass

        tracks.append(TrackInfo(
            element=elem,
            track_id=elem.get("TrackID", ""),
            name=elem.get("Name", ""),
            artist=elem.get("Artist", ""),
            location_url=location_url,
            audio_path=audio_path,
            average_bpm=average_bpm,
            tempos=tempos,
            existing_hot_cue_nums=existing_hot_cue_nums,
        ))

    return tree, tracks


def write_cue_points(track: TrackInfo, cues: list[CuePoint]) -> None:
    """
    Add cue points to a TrackInfo's underlying XML element in-place.

    Appends POSITION_MARK elements for each CuePoint. For each hot cue
    (Num 0-7) a paired memory cue (Num=-1) is also written at the same
    position, which is the standard rekordbox convention.

    Existing POSITION_MARK elements are not removed; call
    clear_hot_cues() first if overwriting.
    """
    elem = track.element
    for cue in cues:
        attrib: dict[str, str] = {
            "Name": cue.name,
            "Type": "0",
            "Start": f"{cue.start:.3f}",
            "Num": str(cue.num),
        }
        if cue.red is not None and cue.green is not None and cue.blue is not None:
            attrib["Red"] = str(cue.red)
            attrib["Green"] = str(cue.green)
            attrib["Blue"] = str(cue.blue)
        elem.append(_stdlib_ET.Element("POSITION_MARK", attrib))

        # Paired memory cue (no colour)
        if cue.num >= 0:
            mem_attrib = {
                "Name": cue.name,
                "Type": "0",
                "Start": f"{cue.start:.3f}",
                "Num": "-1",
            }
            elem.append(_stdlib_ET.Element("POSITION_MARK", mem_attrib))


def clear_hot_cues(track: TrackInfo) -> None:
    """
    Remove all POSITION_MARK elements from the track's XML element.

    This includes both hot cues (Num 0-7) and memory cues (Num -1).
    """
    elem = track.element
    to_remove = elem.findall("POSITION_MARK")
    for pm in to_remove:
        elem.remove(pm)
    track.existing_hot_cue_nums.clear()


def save_xml(tree: _stdlib_ET.ElementTree, output_path: Path) -> None:
    """
    Write the modified ElementTree to disk.

    Preserves the XML declaration and uses UTF-8 encoding.
    """
    _stdlib_ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def iter_tracks(tracks: list[TrackInfo], name_filter: str | None = None) -> Iterator[TrackInfo]:
    """
    Yield tracks, optionally filtered by a case-insensitive substring match
    against the track name or artist.
    """
    for track in tracks:
        if name_filter:
            haystack = f"{track.artist} {track.name}".lower()
            if name_filter.lower() not in haystack:
                continue
        yield track
