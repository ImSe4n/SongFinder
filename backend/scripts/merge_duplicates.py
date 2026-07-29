"""Merge duplicate Song rows created by the old bulk_train.py bug.

Before scripts/bulk_train.py tracked a `source_file` column, the "already
trained?" check matched by title/artist text. That text could change between
runs (e.g. a --no-spotify placeholder name later replaced by real Spotify
metadata), so the same file sometimes got trained multiple times under
separate Song rows — splitting one song's fingerprints/rhythm samples across
several ids and weakening matching for it.

This script finds Song rows that share the same normalized (title, artist),
picks the best-metadata row as canonical, reassigns every other row's
fingerprints and rhythm samples onto it, recomputes its counters, and deletes
the now-empty duplicate rows.

ALWAYS run --dry-run first. This writes directly to the database — make sure
uvicorn and any other script are stopped, and back up songfinder.db first.

Usage (from the backend/ directory):
    .venv/Scripts/python scripts/merge_duplicates.py --dry-run
    .venv/Scripts/python scripts/merge_duplicates.py
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Many titles/artists are non-Latin (K-pop, C-pop, etc.). Windows consoles
# often default stdout to a legacy codepage (cp1252) that can't encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _score(song) -> tuple:
    """Higher is better. Prefer real Spotify metadata over placeholders, then
    richer metadata, then more training data."""
    is_real_spotify = not song.spotify_track_id.startswith("local:")
    return (
        is_real_spotify,
        song.album_art_url is not None,
        bool(song.genres),
        song.release_year is not None,
        song.lyrics is not None,
        song.audio_sample_count,
        song.sample_count,
        -song.id,  # stable tiebreak: prefer the older (lower id) row
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be merged, change nothing"
    )
    args = parser.parse_args()

    from sqlalchemy import func, select, text

    from app.database import Base, SessionLocal, engine, ensure_schema
    from app.models import AudioFingerprint, RhythmSample, Song
    from app.rhythm.features import bucket_tempo

    Base.metadata.create_all(bind=engine)
    ensure_schema()

    with SessionLocal() as db:
        before_songs = db.scalar(select(func.count(Song.id)))
        before_fp = db.scalar(select(func.count(AudioFingerprint.id)))
        before_rs = db.scalar(select(func.count(RhythmSample.id)))

        groups: dict[tuple[str, str], list[Song]] = defaultdict(list)
        for song in db.scalars(select(Song)).all():
            key = (song.title.strip().lower(), song.artist.strip().lower())
            groups[key].append(song)

        duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
        extra_rows = sum(len(v) - 1 for v in duplicate_groups.values())

        print(
            f"{before_songs} songs total, {len(duplicate_groups)} duplicate group(s), "
            f"{extra_rows} extra row(s) to merge away.\n"
        )

        if not duplicate_groups:
            print("Nothing to merge.")
            return 0

        for i, ((title, artist), songs) in enumerate(sorted(duplicate_groups.items()), start=1):
            songs.sort(key=_score, reverse=True)
            winner, losers = songs[0], songs[1:]
            loser_ids = [s.id for s in losers]

            fp_counts = {
                sid: db.scalar(
                    select(func.count(AudioFingerprint.id)).where(AudioFingerprint.song_id == sid)
                )
                for sid in loser_ids
            }
            rs_counts = {
                sid: db.scalar(
                    select(func.count(RhythmSample.id)).where(RhythmSample.song_id == sid)
                )
                for sid in loser_ids
            }
            total_fp_moving = sum(fp_counts.values())
            total_rs_moving = sum(rs_counts.values())

            print(
                f"[{i}/{len(duplicate_groups)}] \"{winner.title}\" — {winner.artist}: "
                f"keep id={winner.id} ({'real' if not winner.spotify_track_id.startswith('local:') else 'placeholder'} "
                f"spotify id), merge {len(losers)} row(s) "
                f"(ids {loser_ids}) — moving {total_fp_moving} fingerprints, {total_rs_moving} rhythm samples"
            )

            if args.dry_run:
                continue

            for loser_id in loser_ids:
                db.execute(
                    text("UPDATE audio_fingerprints SET song_id = :w WHERE song_id = :l"),
                    {"w": winner.id, "l": loser_id},
                )
                db.execute(
                    text("UPDATE rhythm_samples SET song_id = :w WHERE song_id = :l"),
                    {"w": winner.id, "l": loser_id},
                )

            winner.audio_sample_count = winner.audio_sample_count + sum(
                s.audio_sample_count for s in losers
            )
            if winner.lyrics is None:
                winner.lyrics = next((s.lyrics for s in losers if s.lyrics), None)
            if winner.source_file is None:
                winner.source_file = next((s.source_file for s in losers if s.source_file), None)

            for loser in losers:
                db.delete(loser)
            db.flush()

            # Recompute from the merged rhythm samples, same logic as
            # app/services/catalog.record_sample.
            all_samples = db.scalars(
                select(RhythmSample).where(RhythmSample.song_id == winner.id)
            ).all()
            winner.sample_count = sum(1 for s in all_samples if s.source != "auto")
            all_iois = [ioi for s in all_samples for ioi in s.ioi_ms]
            if all_iois:
                median_ioi = statistics.median(all_iois)
                winner.tempo_bpm = 60_000.0 / median_ioi if median_ioi > 0 else None
                winner.tempo_bucket = bucket_tempo(winner.tempo_bpm)

            db.commit()

        if args.dry_run:
            print("\nDry run — nothing was changed. Re-run without --dry-run to apply.")
            return 0

        after_songs = db.scalar(select(func.count(Song.id)))
        after_fp = db.scalar(select(func.count(AudioFingerprint.id)))
        after_rs = db.scalar(select(func.count(RhythmSample.id)))

        print(
            f"\nDone. songs: {before_songs} -> {after_songs} "
            f"(-{before_songs - after_songs}) | "
            f"fingerprints: {before_fp} -> {after_fp} "
            f"({'preserved' if before_fp == after_fp else 'MISMATCH — investigate'}) | "
            f"rhythm_samples: {before_rs} -> {after_rs} "
            f"({'preserved' if before_rs == after_rs else 'MISMATCH — investigate'})"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
