import math

import numpy as np

from app.audio.onsets import extract_tap_times
from tests.audio_utils import tone_melody

# A distinctive long-short rhythm (seconds per note) — what the extractor
# should recover as tap intervals.
RHYTHM_DURATIONS = [0.6, 0.3, 0.3, 0.6, 0.3, 0.6, 0.3, 0.3, 0.6, 0.6]
RHYTHM_ROOTS = [440, 523, 659, 440, 587, 784, 659, 523, 440, 587]


def test_extracts_taps_matching_note_onsets():
    samples = tone_melody(RHYTHM_ROOTS, RHYTHM_DURATIONS)
    taps = extract_tap_times(samples)

    assert len(taps) >= 6
    assert taps[0] == 0

    # The extractor may start at any note (the t=0 attack sits on the STFT
    # edge and is often missed) — so try aligning the tap sequence against
    # the true onset grid re-zeroed at each possible starting note, and
    # require every tap to land near a true onset (±80ms) for the best fit.
    true_onsets = np.cumsum([0.0] + [d * 1000 for d in RHYTHM_DURATIONS[:-1]])

    def alignment_error(start_index: int) -> float:
        rezeroed = true_onsets[start_index:] - true_onsets[start_index]
        return max(min(abs(tap - e) for e in rezeroed) for tap in taps)

    best_error = min(alignment_error(k) for k in range(len(true_onsets) - 2))
    assert best_error < 80, f"taps {taps} don't align with true onsets (err {best_error}ms)"


def test_extracted_rhythm_is_sparse():
    samples = tone_melody(RHYTHM_ROOTS, RHYTHM_DURATIONS)
    taps = extract_tap_times(samples)
    iois = [taps[i + 1] - taps[i] for i in range(len(taps) - 1)]
    assert all(ioi >= 150 for ioi in iois)  # human-like minimum gap


def test_silence_yields_no_taps():
    assert extract_tap_times(np.zeros(11025 * 5, dtype=np.float32)) == []


def test_too_short_input_yields_no_taps():
    assert extract_tap_times(np.zeros(100, dtype=np.float32)) == []


def test_long_input_is_capped_and_windowed():
    # ~36s of audio: extraction should focus on one window and cap tap count
    long_durations = (RHYTHM_DURATIONS * 8)[:80]
    long_roots = (RHYTHM_ROOTS * 8)[:80]
    taps = extract_tap_times(tone_melody(long_roots, long_durations))
    assert 2 <= len(taps) <= 16
    assert taps[-1] <= 16_000  # confined to roughly one 15s window


def test_tempo_invariant_match_between_auto_and_human_taps():
    """The point of the experiment: a human tapping this rhythm (at a
    different tempo) should DTW-match the machine-extracted taps."""
    from app.rhythm.features import iois_to_log_ratio, taps_to_iois
    from app.rhythm.matching import dtw_distance

    auto_taps = extract_tap_times(tone_melody(RHYTHM_ROOTS, RHYTHM_DURATIONS))
    assert len(auto_taps) >= 6

    # a human tapping the same long-short pattern 30% faster
    human_taps = [round(sum(RHYTHM_DURATIONS[:i]) * 1000 / 1.3) for i in range(len(RHYTHM_DURATIONS))]

    d = dtw_distance(
        iois_to_log_ratio(taps_to_iois(auto_taps)),
        iois_to_log_ratio(taps_to_iois(human_taps)),
    )
    assert math.isfinite(d)
    assert d < 0.25, f"auto-vs-human distance too large: {d}"
