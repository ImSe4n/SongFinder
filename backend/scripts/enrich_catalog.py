"""Backfill real Spotify metadata for songs trained with --no-spotify.

Songs created offline carry a placeholder id ("local:...") and no genres,
release year, or album art. This script re-resolves each one via Spotify
(by its title + artist) and updates the row in place — fingerprints, rhythm
samples, and lyrics all stay attached because the row id never changes.

Usage (from the backend/ directory, once Spotify is reachable again):
    .venv/Scripts/python scripts/enrich_catalog.py
    .venv/Scripts/python scripts/enrich_catalog.py --dry-run
"""

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Many trained titles/artists are non-Latin (K-pop, C-pop, etc.). Windows
# consoles often default stdout to a legacy codepage (cp1252) that can't
# encode them, which would otherwise crash an unattended multi-hour run on
# the first such title.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

POLITENESS_DELAY_S = 1.0
MAX_RETRIES = 4
TRANSPORT_RETRY_WAIT_S = 5  # plain network hiccups (timeouts, connection resets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List what would be enriched, change nothing")
    args = parser.parse_args()

    from sqlalchemy import select

    from app.database import Base, SessionLocal, engine, ensure_schema
    from app.models import Song
    from app.services.spotify import get_spotify_client

    Base.metadata.create_all(bind=engine)
    ensure_schema()
    spotify = get_spotify_client()

    with SessionLocal() as db:
        pending = list(
            db.scalars(select(Song).where(Song.spotify_track_id.like("local:%"))).all()
        )
        if not pending:
            print("Nothing to enrich — no offline-trained songs found.")
            return 0

        print(f"Found {len(pending)} offline-trained song(s) to enrich.")
        if args.dry_run:
            for song in pending:
                print(f"  would enrich: {song.title} — {song.artist}")
            return 0

        enriched = failed = skipped = 0
        total = len(pending)
        for index, song in enumerate(pending, start=1):
            query = f"{song.artist} {song.title}" if song.artist != "Unknown" else song.title
            print(f"[{index}/{total}] {song.title} — {song.artist}")
            resolved = None
            gave_up = False
            for attempt in range(MAX_RETRIES):
                try:
                    resolved = spotify.resolve_track(query)
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429 and attempt < MAX_RETRIES - 1:
                        wait = min(max(int(exc.response.headers.get("Retry-After") or 30), 5), 900)
                        print(f"  rate limited — waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    if exc.response.status_code == 503:
                        print("\nSpotify is still blocking this app (503). Try again later.")
                        print(f"Done so far: enriched={enriched} skipped={skipped} failed={failed}")
                        return 1
                    raise
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    # Plain network hiccup — not a rate limit, just retry. This
                    # is what crashed the whole run before: a bare ReadTimeout
                    # wasn't caught here at all.
                    if attempt < MAX_RETRIES - 1:
                        print(f"  network hiccup ({exc.__class__.__name__}) — retrying in {TRANSPORT_RETRY_WAIT_S}s...")
                        time.sleep(TRANSPORT_RETRY_WAIT_S)
                        continue
                    print(f"  FAIL (network): {song.title} — {song.artist}: {exc}")
                    failed += 1
                    gave_up = True

            if gave_up:
                continue
            if resolved is None:
                print(f"  FAIL (not found on Spotify): {song.title} — {song.artist}")
                failed += 1
                continue

            conflict = db.scalar(
                select(Song).where(Song.spotify_track_id == resolved.spotify_track_id)
            )
            if conflict is not None and conflict.id != song.id:
                print(
                    f"  skip: '{song.title} — {song.artist}' resolves to a song already "
                    f"in the catalog (id {conflict.id}) — merge manually if needed"
                )
                skipped += 1
                continue

            song.spotify_track_id = resolved.spotify_track_id
            song.title = resolved.title
            song.artist = resolved.artist
            song.genres = resolved.genres
            song.release_year = resolved.release_year
            song.album_art_url = resolved.album_art_url
            db.commit()
            enriched += 1
            print(f"  enriched -> {song.title} — {song.artist}")
            time.sleep(POLITENESS_DELAY_S)

        print(f"\nDone. enriched={enriched} skipped={skipped} failed={failed}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
