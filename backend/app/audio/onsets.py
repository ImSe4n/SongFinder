"""Experimental: extract a human-like tap sequence from taught audio.

Humans tap the *melody* — a sparse sequence following one line, not the full
mix's every drum hit. This module approximates that: spectral flux restricted
to the melody/vocal frequency band, adaptive-threshold peak picking with a
minimum gap between onsets (forcing human-like sparsity), and — for full-song
inputs — selecting the most energetic ~15s window (usually the hook/chorus)
rather than fingerprinting the whole arrangement.

Extracted sequences are stored as RhythmSample(source='auto') and are
deliberately score-penalized during matching (see AUTO_RHYTHM_DISTANCE_PENALTY)
so they can surface a song as a *candidate* without ever outranking rhythms
real humans taught. If the experiment fails, deleting all source='auto' rows
removes every trace of it.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from app.audio.decode import TARGET_SAMPLE_RATE
from app.config import settings

# Finer time resolution than the fingerprinter: hop 256 @ 11025 Hz ≈ 23 ms.
STFT_WINDOW = 1024
STFT_HOP = 256

# Melody/vocal band: fundamentals + low harmonics live here; excludes most
# kick/bass rumble and hi-hat sizzle.
BAND_LOW_HZ = 150.0
BAND_HIGH_HZ = 2500.0

MIN_GAP_MS = 180  # humans rarely tap faster than this
ONSET_THRESHOLD_FACTOR = 1.4  # peak must exceed local mean flux by this factor
GLOBAL_FLOOR_FRACTION = 0.05  # ...and this fraction of the loudest onset,
# so numerical noise in near-silent stretches can't register as taps
WINDOW_SECONDS = 15.0  # hook-selection window for full-song inputs
MAX_TAPS = 16


def _flux(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Band-limited positive spectral flux per frame, plus seconds-per-frame."""
    freqs, _, stft = signal.stft(
        samples,
        fs=TARGET_SAMPLE_RATE,
        nperseg=STFT_WINDOW,
        noverlap=STFT_WINDOW - STFT_HOP,
        padded=False,
        boundary=None,
    )
    magnitude = np.abs(stft)
    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    banded = magnitude[band]
    diff = np.diff(banded, axis=1)
    flux = np.maximum(diff, 0.0).sum(axis=0)
    # light smoothing so a single noisy frame doesn't split one onset in two
    kernel = np.hanning(5)
    flux = np.convolve(flux, kernel / kernel.sum(), mode="same")
    return flux, STFT_HOP / TARGET_SAMPLE_RATE


def _pick_onsets(flux: np.ndarray, seconds_per_frame: float) -> list[int]:
    """Frame indices of onset peaks: local maxima above an adaptive threshold,
    at least MIN_GAP_MS apart."""
    if flux.size == 0:
        return []
    min_gap_frames = max(1, int((MIN_GAP_MS / 1000) / seconds_per_frame))
    local_mean = uniform_filter1d(flux, size=4 * min_gap_frames + 1)
    threshold = local_mean * ONSET_THRESHOLD_FACTOR + GLOBAL_FLOOR_FRACTION * float(flux.max())

    peaks, _ = signal.find_peaks(flux, height=threshold, distance=min_gap_frames)
    return peaks.tolist()


def extract_tap_times(samples: np.ndarray) -> list[int]:
    """Extract a sparse, human-like tap sequence (ms, first tap at 0).

    Returns [] when no usable rhythm can be extracted — callers should treat
    that as "no auto sample", never an error.
    """
    if samples.size < STFT_WINDOW * 2:
        return []

    flux, spf = _flux(samples)
    onset_frames = _pick_onsets(flux, spf)
    if len(onset_frames) < 2:
        return []

    # Full-song input: keep only the most energetic WINDOW_SECONDS stretch —
    # a proxy for the hook/chorus, the part a human would actually tap.
    total_seconds = flux.size * spf
    if total_seconds > WINDOW_SECONDS * 1.5:
        window_frames = int(WINDOW_SECONDS / spf)
        onset_strengths = np.zeros(flux.size)
        for f in onset_frames:
            onset_strengths[f] = flux[f]
        cumulative = np.concatenate(([0.0], np.cumsum(onset_strengths)))
        starts = np.arange(0, flux.size - window_frames)
        if starts.size > 0:
            window_sums = cumulative[starts + window_frames] - cumulative[starts]
            best_start = int(starts[int(np.argmax(window_sums))])
            onset_frames = [
                f for f in onset_frames if best_start <= f < best_start + window_frames
            ]

    onset_frames = onset_frames[:MAX_TAPS]
    if len(onset_frames) < 2:
        return []

    first = onset_frames[0]
    return [round((f - first) * spf * 1000) for f in onset_frames]


def _select_hook_window_seconds(samples: np.ndarray) -> tuple[float, float]:
    """Same window-selection logic as extract_tap_times' full-song branch,
    exposed standalone so vocal-separated extraction can isolate just this
    stretch (keeps Demucs inference time to seconds, not minutes) instead of
    running separation on the whole track. For inputs already short, returns
    the whole input's time range."""
    total_seconds = samples.size / TARGET_SAMPLE_RATE
    if samples.size < STFT_WINDOW * 2:
        return 0.0, total_seconds

    flux, spf = _flux(samples)
    if flux.size == 0 or total_seconds <= WINDOW_SECONDS * 1.5:
        return 0.0, total_seconds

    onset_frames = _pick_onsets(flux, spf)
    window_frames = int(WINDOW_SECONDS / spf)
    if not onset_frames or window_frames <= 0 or window_frames >= flux.size:
        return 0.0, total_seconds

    onset_strengths = np.zeros(flux.size)
    for f in onset_frames:
        onset_strengths[f] = flux[f]
    cumulative = np.concatenate(([0.0], np.cumsum(onset_strengths)))
    starts = np.arange(0, flux.size - window_frames)
    if starts.size == 0:
        return 0.0, total_seconds
    window_sums = cumulative[starts + window_frames] - cumulative[starts]
    best_start = int(starts[int(np.argmax(window_sums))])
    return best_start * spf, (best_start + window_frames) * spf


def extract_tap_times_vocal_aware(samples: np.ndarray, data: bytes) -> list[int]:
    """Best-effort: isolate vocals in the hook window via Demucs (see
    app/audio/vocal_separation.py), then run the same onset-picking on that
    cleaner signal — much closer to what a human would actually tap, since
    drums/bass/harmony are no longer competing for the same peaks.

    Falls back to raw-mix extract_tap_times (completely unchanged) if vocal
    separation is disabled, unavailable, or fails for any reason — this
    function can never raise and never performs worse than the fallback
    already did.
    """
    if settings.vocal_separation_enabled:
        try:
            from app.audio.vocal_separation import separate_vocals_window

            start_s, end_s = _select_hook_window_seconds(samples)
            vocal_slice = separate_vocals_window(data, start_s, end_s, TARGET_SAMPLE_RATE)
            taps = extract_tap_times(vocal_slice)
            if taps:
                return taps
        except Exception:
            pass
    return extract_tap_times(samples)
