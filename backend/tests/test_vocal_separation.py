"""These actually invoke Demucs (no fake/mock — there's nothing meaningful
to fake here, the whole point is measuring real separation behavior), so
they're slower than the rest of the suite (~10-30s) and download the model
on first run in a fresh environment. Skipped automatically if torch/demucs
aren't installed, so the rest of the suite still runs on a minimal setup."""

import numpy as np
import pytest

from tests.audio_utils import MELODY_A, tone_melody, to_wav_bytes

torch = pytest.importorskip("torch")
pytest.importorskip("demucs")

from app.audio.onsets import _select_hook_window_seconds, extract_tap_times_vocal_aware
from app.audio.vocal_separation import VocalSeparationUnavailable, separate_vocals_window
from app.config import settings
from app.audio.decode import TARGET_SAMPLE_RATE


def test_separate_vocals_window_returns_mono_at_target_rate():
    samples = tone_melody(MELODY_A)
    data = to_wav_bytes(samples)
    duration = samples.size / TARGET_SAMPLE_RATE

    out = separate_vocals_window(data, 0.0, duration, TARGET_SAMPLE_RATE)

    assert out.ndim == 1
    assert out.dtype == np.float32
    # length should roughly match the requested window at target_rate
    assert abs(out.size - samples.size) < TARGET_SAMPLE_RATE  # within 1s tolerance


def test_separate_vocals_window_respects_time_range():
    samples = tone_melody(MELODY_A)
    data = to_wav_bytes(samples)
    full_duration = samples.size / TARGET_SAMPLE_RATE

    half = separate_vocals_window(data, 0.0, full_duration / 2, TARGET_SAMPLE_RATE)
    assert half.size < samples.size


def test_separate_vocals_window_empty_range_raises():
    samples = tone_melody(MELODY_A)
    data = to_wav_bytes(samples)
    with pytest.raises(VocalSeparationUnavailable):
        separate_vocals_window(data, 5.0, 5.0, TARGET_SAMPLE_RATE)


def test_select_hook_window_short_input_returns_whole_range():
    samples = tone_melody(MELODY_A)  # a few seconds, well under WINDOW_SECONDS*1.5
    start, end = _select_hook_window_seconds(samples)
    assert start == 0.0
    assert abs(end - samples.size / TARGET_SAMPLE_RATE) < 1e-6


def test_extract_tap_times_vocal_aware_falls_back_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "vocal_separation_enabled", False)
    samples = tone_melody(MELODY_A)
    data = to_wav_bytes(samples)

    from app.audio.onsets import extract_tap_times

    vocal_aware = extract_tap_times_vocal_aware(samples, data)
    raw = extract_tap_times(samples)
    assert vocal_aware == raw  # disabled -> identical to the raw-mix path


def test_extract_tap_times_vocal_aware_survives_garbage_audio_bytes():
    """Malformed 'audio' bytes must fall back cleanly, never raise."""
    samples = tone_melody(MELODY_A)
    garbage = b"this is not an audio file"
    taps = extract_tap_times_vocal_aware(samples, garbage)
    assert isinstance(taps, list)  # didn't raise; fell back to raw-mix extraction


def test_extract_tap_times_vocal_aware_produces_a_sparse_sequence():
    samples = tone_melody(MELODY_A)
    data = to_wav_bytes(samples)
    taps = extract_tap_times_vocal_aware(samples, data)
    if taps:  # vocal separation may legitimately yield nothing on pure test tones
        assert taps[0] == 0
        iois = [taps[i + 1] - taps[i] for i in range(len(taps) - 1)]
        assert all(ioi >= 150 for ioi in iois)  # human-like minimum gap, same as raw path
