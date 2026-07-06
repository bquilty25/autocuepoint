# autocuepoint

Automatically places hot cue points in your rekordbox 6/7 library by analysing each track's audio and detecting drops, breakdowns, and other major phrase changes.

Cue points are written directly to the rekordbox database (`master.db`), so there is no XML import step. Beat positions are read from rekordbox's own ANLZ analysis files, so every cue lands exactly on the beat grid.

---

## Requirements

- macOS (rekordbox stores its database at `~/Library/Pioneer/rekordbox/master.db`)
- Python 3.10+
- rekordbox 6 or 7 must have been opened at least once to create the database and ANLZ analysis files

## Dependencies

All dependencies are installed automatically by `pip install -e .`.

| Package | Version | Purpose |
|---|---|---|
| [pyrekordbox](https://pyrekordbox.readthedocs.io) | >=0.3 | Reads and writes the rekordbox SQLite database and parses ANLZ beat grid files |
| [librosa](https://librosa.org) | >=0.10 | Audio loading, beat tracking, MFCC/chroma feature extraction, and structural segmentation |
| [numpy](https://numpy.org) | >=1.24 | Array operations used throughout audio and beat-grid processing |
| [scipy](https://scipy.org) | >=1.11 | Required by librosa for signal processing |
| [click](https://click.palletsprojects.com) | >=8.1 | Command-line interface |
| [defusedxml](https://github.com/tiran/defusedxml) | >=0.7 | Safe XML parsing (prevents XML injection attacks when reading rekordbox export files) |

---

## Installation

```bash
git clone https://github.com/your-username/autocuepoint.git
cd autocuepoint
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Usage

**Close rekordbox before running.** The tool creates an automatic backup of `master.db` before writing anything.

```bash
source .venv/bin/activate

# Place cues 16 bars before each drop/phrase change (recommended)
autocuepoint --offset 16

# Place cues 32 bars before each drop/phrase change
autocuepoint --offset 32

# Only process tracks matching a name/artist substring
autocuepoint --offset 16 --track "disclosure"

# Replace existing hot cues
autocuepoint --offset 16 --overwrite

# Fewer cues per track (e.g. intro, drop, breakdown, outro only)
autocuepoint --offset 16 --segments 4

# Chroma mode — better for melody-driven pop/rock
autocuepoint --offset 16 --feature chroma
```

After running, open rekordbox and the cue points will appear on your tracks.

---

## Options

| Flag | Default | Description |
|---|---|---|
| `--offset` | `0` | How many bars to shift each cue before the detected phrase start. Accepts `0`, `16`, or `32`. |
| `--segments` | `8` | Number of phrases to detect per track (max 8, one per hot cue slot). |
| `--overwrite` | off | Replace existing hot cues. Without this flag, tracks that already have cues are skipped. |
| `--track` | — | Case-insensitive substring filter on artist or title. |
| `--feature` | `mfcc` | Analysis method. `mfcc` works best for electronic/dance music; `chroma` works better for melody-driven pop/rock. |
| `--min-duration` | `60` | Skip tracks shorter than this many seconds (useful for filtering out short loops or samples). |
| `--no-backup` | off | Skip the automatic database backup. Not recommended. |

---

## The offset option

The offset shifts each detected phrase boundary backwards by N bars so the cue arrives before the drop rather than on it. When the cue light triggers on your CDJ, you have those bars to line up your mix.

For example, with `--offset 16`:

- A drop is detected at bar 64.
- The cue is placed at bar 48, 16 bars before the drop.
- The cue lights up with 16 bars to go — enough time to start your transition and land cleanly on the drop.

If an offset would push a cue before the start of the track, that cue is skipped. Slot A is always placed at bar 1 regardless of offset.

---

## Beat grid accuracy

For tracks that rekordbox has already analysed, the tool reads exact beat timestamps from the ANLZ `.DAT` files at `~/Library/Pioneer/rekordbox/share/PIONEER/USBANLZ/`. Every cue lands on rekordbox's own beat grid.

For tracks without ANLZ data the tool falls back to rekordbox's stored average BPM, then to audio-based beat detection via librosa.

---

## Hot cue slots

Detected phrase starts are assigned to slots A through H in order.

| Slot | Colour |
|---|---|
| A | Red |
| B | Orange |
| C | Yellow |
| D | Green |
| E | Cyan |
| F | Blue |
| G | Purple |
| H | Pink |

Colours are visible when loading the track in rekordbox's waveform view.

---

## Safety

- A timestamped backup of `master.db` is created before every run: `master.backup_autocue_YYYYMMDD_HHMMSS.db`.
- Nothing is committed to the database unless all cue rows for a track are written successfully.
- Tracks with Spotify streaming URIs are silently skipped (no local audio file to analyse).

---

## Limitations

- macOS only.
- Tracks must have been analysed by rekordbox at least once for ANLZ files to exist. If not, the tool falls back to average BPM stored in the library.
- **If rekordbox's beat grid is wrong, all cues will be off-grid.** The tool trusts the ANLZ beat grid as-is and has no way to detect that it is incorrect. If cues look misaligned, re-analyse the track in rekordbox (right-click > Analyse Tracks), then re-run autocuepoint with `--overwrite`. Tracks that rekordbox struggles to analyse — live recordings, non-4/4 time, unusual intros — are most likely to have grid issues.
- Phrase detection is automatic and unsupervised: it finds points where the audio texture changes noticeably. Results vary by genre and recording style and may not match the musical structure you would identify by ear — treat the cues as a starting point and edit them in rekordbox as needed.
- Cue points are not distributed evenly across the track. The algorithm finds the most structurally distinct boundaries globally, so a track with a repetitive intro and a varied second half will have most cues clustered in the latter half. If you want coverage across the whole track, edit them manually after running.
- Spotify-linked tracks have no local audio file and cannot be processed.

---

## How it works (technical detail)

### 1. Reading the track list

autocuepoint connects to `master.db` (rekordbox's SQLite library database) using [pyrekordbox](https://pyrekordbox.readthedocs.io) and fetches every track that has a stored BPM value.

### 2. Locating beat positions

For each track, the tool looks for an ANLZ `.DAT` file in `~/Library/Pioneer/rekordbox/share/`. These are the analysis files rekordbox writes when you import a track. They contain precise timestamps (in seconds) for every beat, along with a flag for each downbeat (beat 1 of each bar). The tool extracts those downbeat timestamps as the bar grid.

If no ANLZ file exists for a track, the tool falls back to the average BPM stored in the library to estimate bar positions mathematically. If that is also unavailable, it runs [librosa](https://librosa.org)'s beat tracker directly on the audio file.

### 3. Analysing audio texture

The audio file is loaded into memory and two things happen:

- A feature vector is computed for each short audio frame using MFCCs (Mel-Frequency Cepstral Coefficients) — a compact numerical description of the audio's tonal texture at that moment. The `--feature chroma` option uses pitch-class content instead, which works better for melody-driven music.
- Those per-frame values are averaged across each bar to produce one summary vector per bar.

### 4. Detecting phrase boundaries

A self-similarity matrix is built from the bar vectors: each cell scores how similar two bars sound to each other. Agglomerative clustering (from librosa) then divides the matrix into the requested number of phrases (`--segments`) by finding the cut points that best separate dissimilar regions. The first bar of each phrase is returned as a timestamp.

### 5. Applying the offset

Each phrase-start timestamp is shifted backwards by `--offset` bars. This places the cue point before the drop or breakdown rather than on it, so the cue light on your CDJ or controller triggers early enough to act on.

### 6. Writing to the database

The cue timestamps are inserted into the `djmdCue` table in `master.db` as hot cue entries (slots A–H). rekordbox reads these automatically the next time it opens.
