"""Shazam-style audio fingerprinting (Wang 2003).

Pipeline: waveform -> log-magnitude spectrogram -> local-maximum "constellation"
peaks -> anchor/target pair hashes. Each hash packs (anchor_freq, target_freq,
time_delta) into one int and is stored with the anchor's frame time; matching
later works by histogramming the DB-time-minus-query-time offsets of hash hits
(see app/audio/matching.py).

These constants define the *shape* of the algorithm (they change what a stored
hash means), so they are module constants, not .env tunables — changing them
invalidates every stored fingerprint.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter

from app.audio.decode import TARGET_SAMPLE_RATE

# Spectrogram: 2048-sample windows @ 11025 Hz ≈ 186 ms, hop 1024 ≈ 93 ms/frame.
STFT_WINDOW = 2048
STFT_HOP = 1024

# Peak picking: a point is a peak if it's the maximum in a (freq x time)
# neighborhood this large and louder than the given percentile of the whole
# spectrogram (keeps only genuinely prominent energy).
PEAK_NEIGHBORHOOD_FREQ = 25
PEAK_NEIGHBORHOOD_TIME = 11
PEAK_AMP_PERCENTILE = 96.0

# Pairing: each anchor peak pairs with up to FAN_OUT later peaks whose time
# delta is within (MIN_DT, MAX_DT) frames.
FAN_OUT = 8
MIN_DT = 1
MAX_DT = 63  # must fit the 6-bit dt field in the packed hash

_FREQ_BITS = 11  # 2048-point FFT -> 1025 freq bins < 2^11
_DT_BITS = 6


def _spectrogram(samples: np.ndarray) -> np.ndarray:
    _, _, stft = signal.stft(
        samples,
        fs=TARGET_SAMPLE_RATE,
        nperseg=STFT_WINDOW,
        noverlap=STFT_WINDOW - STFT_HOP,
        padded=False,
        boundary=None,
    )
    magnitude = np.abs(stft)
    return np.log(magnitude + 1e-10)


def _find_peaks(spec: np.ndarray) -> list[tuple[int, int]]:
    """Return constellation peaks as (freq_bin, time_frame), time-sorted."""
    local_max = (
        maximum_filter(spec, size=(PEAK_NEIGHBORHOOD_FREQ, PEAK_NEIGHBORHOOD_TIME)) == spec
    )
    threshold = np.percentile(spec, PEAK_AMP_PERCENTILE)
    freq_idx, time_idx = np.nonzero(local_max & (spec > threshold))
    order = np.argsort(time_idx)
    return list(zip(freq_idx[order].tolist(), time_idx[order].tolist()))


def pack_hash(f1: int, f2: int, dt: int) -> int:
    return (f1 << (_FREQ_BITS + _DT_BITS)) | (f2 << _DT_BITS) | dt


def fingerprint(samples: np.ndarray) -> list[tuple[int, int]]:
    """Fingerprint a mono waveform (at TARGET_SAMPLE_RATE).

    Returns a list of (hash, anchor_time_frame) pairs.
    """
    if samples.size < STFT_WINDOW:
        return []

    spec = _spectrogram(samples)
    peaks = _find_peaks(spec)

    hashes: list[tuple[int, int]] = []
    for i, (f1, t1) in enumerate(peaks):
        fanned = 0
        for f2, t2 in peaks[i + 1 :]:
            dt = t2 - t1
            if dt < MIN_DT:
                continue
            if dt > MAX_DT:
                break
            hashes.append((pack_hash(f1, f2, dt), t1))
            fanned += 1
            if fanned >= FAN_OUT:
                break
    return hashes
