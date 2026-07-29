"""Bulk-teach SongFinder's audio engine from a folder of audio files you own.

Every file is decoded, fingerprinted whole (so any part of the song can later
be recognized from a short mic clip), resolved to real metadata via Spotify,
and stored in the catalog — the automated equivalent of teaching songs one at
a time in the app.

File naming: "Artist - Title.mp3" works best (the part before " - " is used
as the artist). Falls back to the whole filename as the search query.
Supported: anything FFmpeg can decode (.mp3, .m4a, .wav, .flac, .ogg, ...).

Usage (from the backend/ directory, with .env configured):
    .venv/Scripts/python scripts/bulk_train.py "C:/path/to/your/music"
    .venv/Scripts/python scripts/bulk_train.py "C:/music" --limit 20 --dry-run
"""

import argparse
import math
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Many resolved titles/artists are non-Latin (K-pop, C-pop, etc.). Windows
# consoles often default stdout to a legacy codepage (cp1252) that can't
# encode them, which would otherwise crash an unattended multi-hour run on
# the first such title.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma"}

# Spotify rate limiting (HTTP 429): wait what Retry-After says, within bounds.
MIN_RETRY_WAIT_S = 5
MAX_RETRY_WAIT_S = 900
MAX_RETRIES_PER_FILE = 4
# Small pause after each Spotify resolution to stay under the rolling window.
POLITENESS_DELAY_S = 1.0


class RateLimitLockout(Exception):
    """Spotify imposed a penalty far too long to sleep through in-process."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        super().__init__(f"rate limited for {seconds}s")


def resolve_with_retry(spotify, query: str) -> tuple[object, float]:
    """resolve_track, sleeping through short 429s. Returns (resolved, seconds_slept).
    Raises RateLimitLockout when Retry-After exceeds the wait cap — at that
    point the whole run should stop rather than poke Spotify per-file."""
    slept = 0.0
    for attempt in range(MAX_RETRIES_PER_FILE):
        try:
            return spotify.resolve_track(query), slept
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429 or attempt == MAX_RETRIES_PER_FILE - 1:
                raise
            retry_after = exc.response.headers.get("Retry-After")
            try:
                wait = int(retry_after) if retry_after else 30
            except ValueError:
                wait = 30
            if wait > MAX_RETRY_WAIT_S:
                raise RateLimitLockout(wait)
            wait = max(wait, MIN_RETRY_WAIT_S)
            print(f"  Spotify rate limit (429) — waiting {format_eta(wait)} then retrying...")
            time.sleep(wait)
            slept += wait
    raise RuntimeError("unreachable")


def format_eta(seconds: float) -> str:
    remaining = max(0, int(math.ceil(seconds)))
    hours, remainder = divmod(remaining, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder containing your audio files")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N files")
    parser.add_argument("--dry-run", action="store_true", help="List what would be trained, change nothing")
    parser.add_argument(
        "--no-spotify",
        action="store_true",
        help="Skip Spotify entirely: take artist/title from the filename "
        "('Artist - Title.mp3') and store a placeholder id. Run "
        "scripts/enrich_catalog.py later to backfill real metadata.",
    )
    parser.add_argument(
        "--no-vocal-separation",
        action="store_true",
        help="Skip Demucs vocal isolation before auto-tap extraction — much "
        "faster (no ~20-30s/song Demucs pass) but the resulting auto-taps "
        "are the older, less accurate raw-mix heuristic (measured at ~0%% "
        "alignment with real human taps — see scripts/eval_auto_rhythm.py). "
        "Use this for a quick first pass over a large library; the auto "
        "samples it creates can be regenerated later by re-teaching those "
        "songs' audio with vocal separation enabled.",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"error: {folder} is not a directory")
        return 1

    files = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("No audio files found.")
        return 1

    print(f"Found {len(files)} audio file(s).")
    if not args.no_vocal_separation and not args.dry_run:
        print(
            "Vocal separation is enabled — each new song gets an extra "
            "Demucs pass (~20-30s on CPU) for auto-tap extraction. Pass "
            "--no-vocal-separation for a much faster (but less accurate) "
            "first pass.\n"
        )
    if args.dry_run:
        for path in files:
            print(f"  would train: {path.name}  (query: \"{path.stem.replace(' - ', ' ')}\")")
        return 0

    # Imports deferred so --dry-run / arg errors don't need the app configured
    from app.audio.decode import AudioDecodeError, decode_to_mono
    from app.audio.fingerprint import fingerprint
    from app.audio.onsets import extract_tap_times, extract_tap_times_vocal_aware
    from app.config import settings
    from app.database import Base, SessionLocal, engine, ensure_schema
    from app.models import Song
    from app.services.catalog import (
        attach_lyrics,
        get_or_create_song,
        record_audio_fingerprints,
        record_sample,
        search_local_catalog,
    )
    from sqlalchemy import select
    from app.services.lyrics import get_lyrics_client
    from app.services.spotify import ResolvedTrack, get_spotify_client

    Base.metadata.create_all(bind=engine)
    ensure_schema()

    spotify = get_spotify_client()
    lyrics = get_lyrics_client()

    def resolve_from_filename(stem: str) -> ResolvedTrack:
        """Offline resolution: 'Artist - Title' from the filename, no Spotify.
        The 'local:' id marks the song for scripts/enrich_catalog.py to
        backfill real metadata later."""
        if " - " in stem:
            artist, title = (part.strip() for part in stem.split(" - ", 1))
        else:
            artist, title = "Unknown", stem.strip()
        return ResolvedTrack(
            spotify_track_id=f"local:{uuid.uuid4().hex}",
            title=title or stem,
            artist=artist or "Unknown",
            genres=[],
            release_year=None,
            album_art_url=None,
        )
    trained = skipped = failed = 0
    started_at = time.monotonic()
    total_slept = 0.0  # 429 waits, excluded from the ETA average

    with SessionLocal() as db:
        for index, path in enumerate(files, start=1):
            query = path.stem.replace(" - ", " ").strip()
            label = path.name
            source_key = str(path.resolve().relative_to(folder.resolve()))
            elapsed = time.monotonic() - started_at - total_slept
            average_seconds = elapsed / (index - 1) if index > 1 else None
            remaining_count = len(files) - index + 1
            eta = format_eta(average_seconds * remaining_count) if average_seconds else "calculating..."
            print(f"[{index}/{len(files)}] eta {eta} - {label}")
            try:
                # Keyed by file, not by title/artist text: text can (and will)
                # change once --no-spotify placeholders get enriched with real
                # Spotify metadata, which would otherwise make this file look
                # untrained and cause a duplicate.
                existing = db.scalar(select(Song).where(Song.source_file == source_key))
                if existing is None:
                    existing = search_local_catalog(db, query)
                if existing is not None and existing.audio_sample_count > 0:
                    if existing.source_file != source_key:
                        existing.source_file = source_key
                        db.commit()
                    print("  skip (already trained)")
                    skipped += 1
                    continue

                raw_bytes = path.read_bytes()
                samples = decode_to_mono(raw_bytes)
                hashes = fingerprint(samples)
                if not hashes:
                    print(f"  FAIL (no audio features): {label}")
                    failed += 1
                    continue

                song = existing
                if song is None:
                    if args.no_spotify:
                        resolved = resolve_from_filename(path.stem)
                    else:
                        try:
                            resolved, slept = resolve_with_retry(spotify, query)
                        except RateLimitLockout as exc:
                            print(
                                f"\nSpotify imposed a long rate-limit penalty "
                                f"(Retry-After {format_eta(exc.seconds)}). Stopping the "
                                f"run — resume after that, or re-run now with "
                                f"--no-spotify and enrich later."
                            )
                            break
                        except httpx.HTTPStatusError as exc:
                            if exc.response.status_code == 503:
                                print(
                                    "\nSpotify is refusing this app's credentials "
                                    "(503 temporarily_unavailable — an extended "
                                    "rate-limit block). Stopping so retries don't "
                                    "extend it. Re-run with --no-spotify to keep "
                                    "training without metadata, then run "
                                    "scripts/enrich_catalog.py once the block lifts."
                                )
                                break
                            raise
                        total_slept += slept
                        time.sleep(POLITENESS_DELAY_S)
                        total_slept += POLITENESS_DELAY_S
                    if resolved is None:
                        print(f"  FAIL (not found on Spotify): {label}")
                        failed += 1
                        continue
                    song = get_or_create_song(db, resolved)

                if song.source_file != source_key:
                    song.source_file = source_key
                    db.commit()

                record_audio_fingerprints(db, song, hashes)

                auto_taps: list[int] = []
                if settings.auto_rhythm_enabled:
                    if args.no_vocal_separation:
                        auto_taps = extract_tap_times(samples)
                    else:
                        auto_taps = extract_tap_times_vocal_aware(samples, raw_bytes)
                    if len(auto_taps) >= settings.min_taps_for_training:
                        record_sample(db, song, auto_taps, source="auto")
                    else:
                        auto_taps = []

                attach_lyrics(db, song, lyrics)
                db.commit()
                trained += 1
                rhythm_note = f", {len(auto_taps)} auto-taps" if auto_taps else ""
                print(f"  trained: {song.title} — {song.artist}  ({len(hashes)} hashes{rhythm_note})")
            except AudioDecodeError as exc:
                db.rollback()
                failed += 1
                print(f"  FAIL (decode): {exc}")
            except Exception as exc:  # keep going on any single-file failure
                db.rollback()
                failed += 1
                print(f"  FAIL: {exc}")

    print(f"\nDone. trained={trained} skipped={skipped} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
