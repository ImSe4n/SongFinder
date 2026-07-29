import math
import statistics

from app.config import settings


def taps_to_iois(taps_ms: list[int]) -> list[int]:
    """Inter-onset intervals: the gap between each consecutive pair of taps."""
    return [taps_ms[i + 1] - taps_ms[i] for i in range(len(taps_ms) - 1)]


def iois_to_log_ratio(iois_ms: list[int], min_ioi_ms: int | None = None) -> list[float]:
    """Tempo-invariant representation: log(ioi[i+1] / ioi[i]) for each consecutive
    pair of intervals. Invariant to uniformly speeding up/slowing down the whole
    rhythm, and symmetric (speeding up vs. slowing down give equal-magnitude,
    opposite-sign values).

    Tiny IOIs (e.g. an accidental double-tap) are floored to `min_ioi_ms` before
    taking the ratio, so they can't blow up the log ratio towards +/-infinity.
    """
    floor = min_ioi_ms if min_ioi_ms is not None else settings.min_ioi_ms
    clamped = [max(ioi, floor) for ioi in iois_ms]
    return [math.log(clamped[i + 1] / clamped[i]) for i in range(len(clamped) - 1)]


def estimate_tempo_bpm(iois_ms: list[int]) -> float | None:
    """Rough tempo estimate from the median inter-onset interval."""
    if not iois_ms:
        return None
    median_ioi = statistics.median(iois_ms)
    if median_ioi <= 0:
        return None
    return 60_000.0 / median_ioi


def bucket_tempo(bpm: float | None) -> str | None:
    if bpm is None:
        return None
    if bpm <= settings.tempo_slow_max_bpm:
        return "slow"
    if bpm >= settings.tempo_fast_min_bpm:
        return "fast"
    return "medium"
