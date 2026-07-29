"""Import real human tap-timing data from the QBT-Extended research dataset
(Kaneshiro, Kim, Herrera, Oh, Berger, Slaney — ISMIR 2013, Stanford CCRMA)
into SongFinder's rhythm-matching corpus.

This is the actual prerequisite our own in-app usage can't provide fast
enough: real people tapping the rhythm of known pop songs' hooks, with
repeat coverage per song (multiple people per song — the "diversity" metric
scripts/training_status.py tracks). ~3,300 tapped queries across 51 famous
pop/rock songs from 60 participants, recorded via touchscreen taps.

*** LICENSE: CC BY-NC-SA 3.0 US — NON-COMMERCIAL USE ONLY. ***
If SongFinder is or becomes a commercial product, this data (and the
`external_qbte` rhythm samples it creates) must be removed, or you need
explicit permission from the dataset authors. Every imported sample is
tagged source='external_qbte' specifically so it can be found and deleted
in one query:
    DELETE FROM rhythm_samples WHERE source = 'external_qbte';
This script refuses to write anything until you pass --i-accept-nc-license.

Get the dataset first (not redistributed here):
    https://ccrma.stanford.edu/groups/qbtextended/
    -> download "SQLite3 database", unzip, note the path to db/qbt.db

Data characteristics worth knowing before you rely on this:
- Two conditions: with_music=1 (tapped ALONG to real playback — precise,
  audio-cued) and with_music=0 (tapped from MEMORY, no audio — closer to
  how SongFinder's own Tap/Training flows actually work, but noisier).
  Both are imported and tagged identically; filter by condition later if
  eval_auto_rhythm-style comparison shows a quality difference worth it.
- Sequences are LONG relative to typical in-app taps (median 33 taps over
  a ~15s excerpt, vs. ~4-11 for our organic samples) — imported whole,
  not truncated, to keep the ground truth authentic. Whether DTW matching
  quality holds up against much shorter real queries is an open question
  to check empirically (scripts/eval_auto_rhythm.py-style validation).
- 2 of 60 participants are flagged arrhythmic/tone-deaf in the dataset's
  own questionnaire data; excluded by default (--include-flagged-users
  to keep them anyway).

Usage (from the backend/ directory):
    .venv/Scripts/python scripts/import_qbt_extended.py path/to/qbt.db --dry-run
    .venv/Scripts/python scripts/import_qbt_extended.py path/to/qbt.db --i-accept-nc-license
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

POLITENESS_DELAY_S = 1.0
SOURCE_TAG = "external_qbte"


def parse_taps_ms(tap_data: str, min_taps: int, max_taps: int, max_span_ms: int) -> list[int] | None:
    """QBT-Extended's tap_data is a comma-separated list of tap-on times in
    SECONDS, absolute within the excerpt session. Convert to our schema's
    convention: milliseconds, relative to the first tap (taps[0] == 0),
    strictly increasing. Returns None if the sequence can't be made valid
    (too short, spans too long, or collapses to <2 distinct points after
    rounding to the millisecond)."""
    try:
        seconds = [float(x) for x in tap_data.split(",") if x.strip()]
    except ValueError:
        return None
    if len(seconds) < min_taps:
        return None

    ms = sorted(round(s * 1000) for s in seconds)
    first = ms[0]
    relative = [t - first for t in ms]

    # de-duplicate: two taps rounding to the same ms would violate the
    # strictly-increasing rule our schema requires
    deduped = [relative[0]]
    for t in relative[1:]:
        if t > deduped[-1]:
            deduped.append(t)

    if len(deduped) < min_taps or len(deduped) > max_taps:
        return None
    if deduped[-1] > max_span_ms:
        return None
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("db_path", help="Path to QBT-Extended's db/qbt.db")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be imported, change nothing")
    parser.add_argument(
        "--i-accept-nc-license",
        action="store_true",
        help="Required to actually write data (see the license notice in this file's docstring)",
    )
    parser.add_argument(
        "--include-flagged-users",
        action="store_true",
        help="Include the 2 participants the dataset's own questionnaire flags as arrhythmic/tone-deaf",
    )
    parser.add_argument("--limit-songs", type=int, default=None, help="Stop after N songs (for testing)")
    args = parser.parse_args()

    src_path = Path(args.db_path)
    if not src_path.is_file():
        print(f"error: {src_path} not found")
        return 1

    if not args.dry_run and not args.i_accept_nc_license:
        print(
            "QBT-Extended is licensed CC BY-NC-SA 3.0 (non-commercial only).\n"
            "Pass --i-accept-nc-license to actually import (or --dry-run to preview).\n"
            "See this script's module docstring for details and the deletion query\n"
            "to remove this data later if needed."
        )
        return 1

    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row

    excluded_users: set[str] = set()
    if not args.include_flagged_users:
        rows = src.execute(
            "select user_id from users where arrhythmic != 0 or tone_deaf != 0"
        ).fetchall()
        excluded_users = {r["user_id"] for r in rows}

    songs_rows = src.execute("select * from songs order by song_title").fetchall()
    if args.limit_songs:
        songs_rows = songs_rows[: args.limit_songs]

    print(f"{len(songs_rows)} song(s) in the dataset, {len(excluded_users)} participant(s) excluded by quality flag.\n")

    if args.dry_run:
        from app.config import settings

        total_valid = 0
        for song_row in songs_rows:
            task_rows = src.execute(
                "select tap_data, user_id from tasks where song_title = ? and tap_data is not null and tap_data != ''",
                (song_row["song_title"],),
            ).fetchall()
            valid = 0
            for t in task_rows:
                if t["user_id"] in excluded_users:
                    continue
                if parse_taps_ms(
                    t["tap_data"],
                    settings.min_taps_for_training,
                    settings.max_tap_count,
                    settings.max_tap_span_ms,
                ):
                    valid += 1
            total_valid += valid
            print(f"  {song_row['song_title']} — {song_row['artist']}: {valid} usable tap sequence(s) of {len(task_rows)}")
        print(f"\nTotal usable sequences: {total_valid}. Re-run with --i-accept-nc-license to import.")
        return 0

    # Imports deferred so --dry-run / arg errors don't need the app configured
    import httpx

    from app.config import settings
    from app.database import Base, SessionLocal, engine, ensure_schema
    from app.services.catalog import get_or_create_song, record_sample, search_local_catalog
    from app.services.spotify import get_spotify_client

    Base.metadata.create_all(bind=engine)
    ensure_schema()
    spotify = get_spotify_client()

    songs_imported = songs_skipped = songs_failed = 0
    samples_imported = samples_skipped_invalid = samples_skipped_user = 0

    with SessionLocal() as db:
        for i, song_row in enumerate(songs_rows, start=1):
            title, artist = song_row["song_title"], song_row["artist"]
            query = f"{artist} {title}"
            print(f"[{i}/{len(songs_rows)}] {title} — {artist}")

            task_rows = src.execute(
                "select tap_data, user_id from tasks where song_title = ? and tap_data is not null and tap_data != ''",
                (title,),
            ).fetchall()

            valid_taps: list[list[int]] = []
            for t in task_rows:
                if t["user_id"] in excluded_users:
                    samples_skipped_user += 1
                    continue
                taps = parse_taps_ms(
                    t["tap_data"], settings.min_taps_for_training, settings.max_tap_count, settings.max_tap_span_ms
                )
                if taps is None:
                    samples_skipped_invalid += 1
                    continue
                valid_taps.append(taps)

            if not valid_taps:
                print("  skip (no usable tap sequences)")
                songs_skipped += 1
                continue

            try:
                song = search_local_catalog(db, query)
                if song is None:
                    resolved = spotify.resolve_track(query)
                    if resolved is None:
                        print(f"  FAIL (not found on Spotify): {query}")
                        songs_failed += 1
                        continue
                    song = get_or_create_song(db, resolved)
                    time.sleep(POLITENESS_DELAY_S)

                for taps in valid_taps:
                    record_sample(db, song, taps, source=SOURCE_TAG)
                db.commit()
                samples_imported += len(valid_taps)
                songs_imported += 1
                print(f"  imported {len(valid_taps)} tap sequence(s) -> {song.title} — {song.artist}")
            except httpx.HTTPError as exc:
                db.rollback()
                songs_failed += 1
                print(f"  FAIL (network): {exc}")
            except Exception as exc:
                db.rollback()
                songs_failed += 1
                print(f"  FAIL: {exc}")

    print(
        f"\nDone. songs: imported={songs_imported} skipped={songs_skipped} failed={songs_failed} | "
        f"samples: imported={samples_imported} skipped_invalid={samples_skipped_invalid} "
        f"skipped_flagged_user={samples_skipped_user}"
    )
    print(f"\nAll imported samples are tagged source='{SOURCE_TAG}'. To remove them later:")
    print(f"  sqlite3 songfinder.db \"DELETE FROM rhythm_samples WHERE source = '{SOURCE_TAG}';\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
