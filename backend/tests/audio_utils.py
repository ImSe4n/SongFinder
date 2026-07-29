"""Synthetic audio for fingerprint tests: a 'melody' of two-note chords is
distinctive enough to fingerprint reliably while being fully deterministic."""

import io
import wave

import numpy as np

from app.audio.decode import TARGET_SAMPLE_RATE

# Chord roots (Hz) — an arbitrary but distinctive up-and-down contour
MELODY_A = [440, 523, 659, 784, 659, 880, 523, 440, 698, 587, 784, 659, 440, 587]
MELODY_B = [330, 415, 494, 370, 554, 311, 466, 415, 622, 494, 349, 415, 554, 330]

SEGMENT_SECONDS = 0.45


def tone_melody(
    roots: list[int],
    segment_seconds: float | list[float] = SEGMENT_SECONDS,
    rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Each segment is a root+fifth chord with a soft attack/decay envelope.
    `segment_seconds` may be a list (one duration per note) to build melodies
    with a distinctive rhythm rather than a steady pulse."""
    durations = (
        segment_seconds
        if isinstance(segment_seconds, list)
        else [segment_seconds] * len(roots)
    )
    segments = []
    for root, seconds in zip(roots, durations):
        n = int(seconds * rate)
        t = np.arange(n) / rate
        envelope = np.minimum(1.0, np.minimum(t / 0.02, (seconds - t) / 0.05))
        chord = 0.6 * np.sin(2 * np.pi * root * t) + 0.4 * np.sin(2 * np.pi * root * 1.5 * t)
        segments.append((chord * envelope).astype(np.float32))
    return np.concatenate(segments)


def clip(samples: np.ndarray, start_s: float, end_s: float, rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    return samples[int(start_s * rate) : int(end_s * rate)]


def to_wav_bytes(samples: np.ndarray, rate: int = TARGET_SAMPLE_RATE) -> bytes:
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()
