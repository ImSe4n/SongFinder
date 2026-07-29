import statistics
from collections import defaultdict

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, selectinload

from app.models import AudioFingerprint, RhythmSample, Song
from app.rhythm.features import bucket_tempo, iois_to_log_ratio, taps_to_iois
from app.services.query_matching import normalize_query, query_variants
from app.services.spotify import ResolvedTrack

# SQLite caps the number of bound parameters per statement; chunk IN() lookups.
_HASH_LOOKUP_CHUNK = 500

# A substring needle shorter than this can match almost anything by chance
# (e.g. "me" inside "american") — independent guard alongside
# MIN_QUERY_PART_LENGTH in query_matching.py, since a short needle could in
# principle reach here some other way, not just via structured_query_parts.
MIN_SUBSTRING_NEEDLE_LENGTH = 4


def _normalize(s: str) -> str:
    return normalize_query(s)


def search_local_catalog(db: Session, query: str) -> Song | None:
    """Conservative local match: normalized substring match on "title artist",
    preferring the song with the most existing samples on ties. Deliberately
    strict (no fuzzy matching) to avoid false-merging distinct songs (e.g.
    a studio cut vs. a live version)."""
    needles = query_variants(query)
    if not needles:
        return None

    candidates = db.scalars(select(Song)).all()
    best_song: Song | None = None
    best_score = 0
    best_sample_count = -1

    for song in candidates:
        haystack = _normalize(f"{song.title} {song.artist}")
        score = 0
        for needle in needles:
            if needle == haystack:
                score = 3
                break
            if len(needle) < MIN_SUBSTRING_NEEDLE_LENGTH:
                continue
            if needle in haystack or haystack in needle:
                score = max(score, 1)

        if score == 0:
            continue

        if score > best_score or (score == best_score and song.sample_count > best_sample_count):
            best_song = song
            best_score = score
            best_sample_count = song.sample_count

    return best_song


def get_or_create_song(db: Session, resolved: ResolvedTrack) -> Song:
    existing = db.scalar(
        select(Song).where(Song.spotify_track_id == resolved.spotify_track_id)
    )
    if existing:
        return existing

    song = Song(
        spotify_track_id=resolved.spotify_track_id,
        title=resolved.title,
        artist=resolved.artist,
        genres=resolved.genres,
        release_year=resolved.release_year,
        album_art_url=resolved.album_art_url,
        sample_count=0,
    )
    db.add(song)
    db.flush()
    return song


def record_sample(db: Session, song: Song, taps_ms: list[int], source: str) -> RhythmSample:
    iois = taps_to_iois(taps_ms)
    log_ratio = iois_to_log_ratio(iois)

    sample = RhythmSample(
        song_id=song.id,
        raw_taps_ms=taps_ms,
        ioi_ms=iois,
        log_ratio_vector=log_ratio,
        source=source,
    )
    db.add(sample)

    # sample_count is the user-facing "how well humans have trained this" —
    # machine-extracted 'auto' samples don't count toward it.
    if source != "auto":
        song.sample_count += 1
    all_iois = [ioi for s in song.samples for ioi in s.ioi_ms] + iois
    if all_iois:
        median_ioi = statistics.median(all_iois)
        tempo_bpm = 60_000.0 / median_ioi if median_ioi > 0 else None
        song.tempo_bpm = tempo_bpm
        song.tempo_bucket = bucket_tempo(tempo_bpm)

    db.flush()
    return sample


def list_matchable_songs(db: Session) -> list[Song]:
    return list(
        db.scalars(select(Song).options(selectinload(Song.samples))).all()
    )


def attach_lyrics(db: Session, song: Song, lyrics_client) -> None:
    """Best-effort: fetch and store full lyrics the first time a song enters
    the catalog, so lyric-fragment search can run locally. Never fatal."""
    if song.lyrics is not None:
        return
    try:
        hit = lyrics_client.get_lyrics(song.title, song.artist)
    except Exception:
        return
    if hit is not None and hit.plain_lyrics:
        song.lyrics = hit.plain_lyrics
        db.flush()


def record_audio_fingerprints(
    db: Session, song: Song, hashes: list[tuple[int, int]]
) -> int:
    """Bulk-insert one taught clip's (hash, offset) pairs for `song`."""
    db.bulk_save_objects(
        [AudioFingerprint(song_id=song.id, hash=h, offset=t) for h, t in hashes]
    )
    song.audio_sample_count += 1
    db.flush()
    return len(hashes)


# How many hash buckets to sample when warming the fingerprint cache — large
# enough to touch a broad, scattered swath of the index's B-tree pages (which
# is what a real query's `hash IN (...)` does), small enough to stay well
# under a second even cold.
_WARM_SAMPLE_HASHES = 500


def warm_fingerprint_cache(db: Session) -> None:
    """Touch a scattered sample of the audio_fingerprints hash index so the OS
    page cache is warm before the first real /audio/query. On a
    multi-million-row table, the very first lookup after process start pays a
    one-time disk-read cost (measured: ~7s cold vs ~1s warm on a 34M-row
    table) — this pays that cost at startup instead of on a user's first
    request. A single sequential `LIMIT` only warms one page, and a full
    COUNT(*) takes 30s+, so instead this samples real hash values spread
    across the table (via a modulo on rowid) and does one `IN (...)` lookup,
    mirroring what a real query actually does. Best-effort; a failure here
    must never take the server down."""
    try:
        sample = db.execute(
            text(
                "SELECT hash FROM audio_fingerprints WHERE rowid % 5000 = 0 "
                "LIMIT :n"
            ),
            {"n": _WARM_SAMPLE_HASHES},
        ).scalars().all()
        if sample:
            db.execute(
                select(AudioFingerprint.hash).where(AudioFingerprint.hash.in_(sample))
            ).all()
    except Exception:
        pass


# A fingerprint hash shared across an enormous number of stored rows (a
# generic spectral peak that just happens to recur constantly) barely
# discriminates one song from another — but fetching it still costs real I/O
# and dominates query time as the catalog grows (measured: a handful of
# hashes accounted for the majority of rows in a single query). Once a hash
# is this common in the catalog, skip it entirely rather than pay to fetch
# rows that won't meaningfully affect ranking.
_HASH_POPULARITY_CAP = 200


def lookup_fingerprints(
    db: Session, hashes: list[int]
) -> dict[int, list[tuple[int, int]]]:
    """Map each query hash to every stored (song_id, offset) with that hash.
    Hashes that are too common in the catalog (see _HASH_POPULARITY_CAP) are
    skipped — they're noise, not signal, for matching purposes."""
    hits: dict[int, list[tuple[int, int]]] = defaultdict(list)
    unique = list(set(hashes))
    for i in range(0, len(unique), _HASH_LOOKUP_CHUNK):
        chunk = unique[i : i + _HASH_LOOKUP_CHUNK]
        selective_hashes = (
            select(AudioFingerprint.hash)
            .where(AudioFingerprint.hash.in_(chunk))
            .group_by(AudioFingerprint.hash)
            .having(func.count() <= _HASH_POPULARITY_CAP)
        )
        rows = db.execute(
            select(
                AudioFingerprint.hash, AudioFingerprint.song_id, AudioFingerprint.offset
            ).where(AudioFingerprint.hash.in_(selective_hashes))
        ).all()
        for h, song_id, offset in rows:
            hits[h].append((song_id, offset))
    return hits
