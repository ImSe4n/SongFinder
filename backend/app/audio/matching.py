"""Matching fingerprints against the stored corpus.

A query hash that also appears in a stored song votes for (song_id, offset)
where offset = stored_frame_time - query_frame_time. A true match produces a
large spike at one consistent offset; random hash collisions scatter across
offsets. The per-song score is the height of its tallest offset bin.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

from app.config import settings


@dataclass
class AudioMatch:
    song_id: int
    aligned_hashes: int
    confidence: float


def confidence_from_aligned(aligned: int) -> float:
    """Map an aligned-hash count to a 0-100 display confidence.
    Saturating curve: ~40 at 15 hashes, ~63 at 30, ~90 at 70."""
    return min(100.0, 100.0 * (1.0 - math.exp(-aligned / 30.0)))


def rank_audio_matches(
    query_hashes: list[tuple[int, int]],
    db_hits: dict[int, list[tuple[int, int]]],
) -> list[AudioMatch]:
    """`query_hashes` is [(hash, query_time)], `db_hits` maps hash ->
    [(song_id, stored_time), ...] for every query hash found in the DB."""
    bins: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for h, t_query in query_hashes:
        for song_id, t_db in db_hits.get(h, ()):
            bins[song_id][t_db - t_query] += 1

    # STFT framing can shift a peak's time by a frame or two between the
    # stored clip and the query — different clip boundaries, decoder
    # resampling, and (per real query-log evidence, 2026-07-22) real phone-mic
    # recordings of speaker playback drift more than a clean re-encode would.
    # A true match's votes get split across adjacent offsets as a result.
    # Score each offset together with its neighbors within OFFSET_TOLERANCE
    # frames (~93ms/frame) rather than just ±1 — still tight enough that an
    # unrelated song's scattered collisions won't accumulate into one bin.
    OFFSET_TOLERANCE = 2
    best_per_song: dict[int, int] = {}
    for song_id, offsets in bins.items():
        best_per_song[song_id] = max(
            sum(offsets.get(o + d, 0) for d in range(-OFFSET_TOLERANCE, OFFSET_TOLERANCE + 1))
            for o in list(offsets)
        )

    ranked = [
        AudioMatch(song_id=sid, aligned_hashes=n, confidence=confidence_from_aligned(n))
        for sid, n in best_per_song.items()
    ]
    ranked.sort(key=lambda m: m.aligned_hashes, reverse=True)
    return ranked


def decide_audio_match(ranked: list[AudioMatch]) -> tuple[str, AudioMatch | None]:
    """Returns ('confident'|'ambiguous'|'no_match', top_or_None)."""
    if not ranked:
        return "no_match", None

    top = ranked[0]
    if top.aligned_hashes < settings.audio_match_min_hashes:
        # Too few aligned hashes to trust at all — likely noise/collisions.
        return ("no_match", None) if top.aligned_hashes < settings.audio_noise_floor_hashes else ("ambiguous", top)

    runner_up = ranked[1].aligned_hashes if len(ranked) > 1 else 0
    if top.aligned_hashes >= settings.audio_match_margin_ratio * max(1, runner_up):
        return "confident", top
    return "ambiguous", top
