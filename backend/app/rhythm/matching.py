import math
from dataclasses import dataclass

from app.config import settings


def dtw_distance(
    a: list[float],
    b: list[float],
    band_fraction: float | None = None,
    cost_cap: float | None = None,
) -> float:
    """Banded, cost-capped, length-normalized DTW distance between two
    tempo-invariant log-ratio sequences.

    - Banded: a Sakoe-Chiba window whose width scales with sequence length
      (not fixed) keeps the alignment sane and the computation bounded.
    - Cost-capped: a single wild outlier tap (fat-finger double-tap, missed
      beat) can't dominate the whole distance.
    - Length-normalized: dividing by max(len(a), len(b)) makes distances
      comparable across samples/songs with different tap counts.
    """
    band_fraction = settings.dtw_band_fraction if band_fraction is None else band_fraction
    cost_cap = settings.dtw_cost_cap if cost_cap is None else cost_cap

    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return cost_cap

    band = max(2, round(band_fraction * max(n, m)))

    inf = math.inf
    prev = [inf] * (m + 1)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr = [inf] * (m + 1)
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            cost = min((a[i - 1] - b[j - 1]) ** 2, cost_cap)
            curr[j] = cost + min(prev[j], curr[j - 1] if j > lo else inf, prev[j - 1])
        prev = curr

    raw = prev[m]
    if math.isinf(raw):
        return cost_cap * max(n, m)
    return raw / max(n, m)


def subsequence_dtw_distance(
    a: list[float],
    b: list[float],
    cost_cap: float | None = None,
) -> float:
    """Open-begin/open-end DTW: aligns the SHORTER sequence against the
    best-matching contiguous region of the longer one, instead of forcing a
    global end-to-end alignment.

    Why this exists: global DTW structurally fails when a short query meets a
    long reference — a real user's ~12-tap query against a ~33-tap stored
    sample racks up huge cost explaining the reference's unmatched tail, no
    matter how well the tapped fragment matches (measured at 0-3% top-1 on
    exactly this condition — see scripts/eval_matchers.py). Here the longer
    sequence's start and end are free: cost is charged only for the region
    the shorter sequence actually aligns to, normalized by the shorter
    length.

    Trade-off to be aware of: free endpoints make matching *easier* for
    everything — a short fragment can find a decent-looking home inside many
    references — so this buys short-query recall at some precision risk.
    That trade is what the eval script quantifies.

    No Sakoe-Chiba band: the band presumes a near-diagonal global alignment,
    which is exactly what this variant abandons. Sequences here are short
    (<=~72 elements), so the full O(n*m) grid stays cheap.
    """
    cost_cap = settings.dtw_cost_cap if cost_cap is None else cost_cap

    query, ref = (a, b) if len(a) <= len(b) else (b, a)
    n, m = len(query), len(ref)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return cost_cap

    inf = math.inf
    # Free start: before consuming any query element, standing anywhere in
    # the reference costs nothing.
    prev = [0.0] * (m + 1)

    for i in range(1, n + 1):
        curr = [inf] * (m + 1)
        for j in range(1, m + 1):
            cost = min((query[i - 1] - ref[j - 1]) ** 2, cost_cap)
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    # Free end: the query may finish anywhere in the reference.
    return min(prev[1:]) / n


def confidence_from_distance(distance: float, alpha: float | None = None) -> float:
    alpha = settings.dtw_alpha if alpha is None else alpha
    confidence = 100.0 * math.exp(-alpha * distance)
    return max(0.0, min(100.0, confidence))


@dataclass
class RankedSong:
    song_id: int
    distance: float
    confidence: float


def rank_candidates(
    query_r: list[float],
    songs_with_vectors: list[tuple[int, list[list[float]], list[list[float]]]],
) -> list[RankedSong]:
    """`songs_with_vectors` is a list of
    (song_id, [human_sample_vector, ...], [auto_sample_vector, ...]).

    Each song's score is its single best-matching (min-distance) sample, but
    machine-extracted 'auto' samples carry a distance penalty: they exist to
    surface a song as a candidate, never to outrank human-taught rhythms.
    """
    penalty = settings.auto_rhythm_distance_penalty
    ranked: list[RankedSong] = []
    for song_id, human_vectors, auto_vectors in songs_with_vectors:
        distances: list[float] = [dtw_distance(query_r, v) for v in human_vectors]
        distances += [dtw_distance(query_r, v) + penalty for v in auto_vectors]
        if not distances:
            continue
        best_distance = min(distances)
        ranked.append(
            RankedSong(
                song_id=song_id,
                distance=best_distance,
                confidence=confidence_from_distance(best_distance),
            )
        )
    ranked.sort(key=lambda r: r.distance)
    return ranked


def decide_match(
    ranked: list[RankedSong],
    threshold: float | None = None,
    margin: float | None = None,
) -> tuple[str, RankedSong | None]:
    """Returns ('confident'|'ambiguous'|'no_match', top_or_None)."""
    threshold = settings.match_confidence_threshold if threshold is None else threshold
    margin = settings.match_margin_threshold if margin is None else margin

    if not ranked:
        return "no_match", None

    top = ranked[0]
    if top.confidence < threshold:
        return "ambiguous", top

    runner_up_confidence = ranked[1].confidence if len(ranked) > 1 else 0.0
    if top.confidence - runner_up_confidence >= margin:
        return "confident", top

    return "ambiguous", top
