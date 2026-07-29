import math

from app.rhythm.features import (
    bucket_tempo,
    estimate_tempo_bpm,
    iois_to_log_ratio,
    taps_to_iois,
)


def test_taps_to_iois():
    assert taps_to_iois([0, 200, 500, 900]) == [200, 300, 400]


def test_taps_to_iois_single_tap_has_no_intervals():
    assert taps_to_iois([0]) == []


def test_log_ratio_basic():
    # constant IOIs -> ratio 1 -> log(1) = 0 throughout
    r = iois_to_log_ratio([200, 200, 200])
    assert all(abs(x) < 1e-9 for x in r)


def test_log_ratio_is_tempo_scale_invariant():
    iois = [200, 300, 150, 400]
    scaled = [i * 2 for i in iois]
    r1 = iois_to_log_ratio(iois)
    r2 = iois_to_log_ratio(scaled)
    for x, y in zip(r1, r2):
        assert math.isclose(x, y, abs_tol=1e-9)


def test_log_ratio_floors_tiny_iois():
    # a near-zero IOI (accidental double tap) must not blow up the log ratio
    r = iois_to_log_ratio([200, 1, 200], min_ioi_ms=40)
    assert all(math.isfinite(x) for x in r)


def test_estimate_tempo_bpm():
    # median IOI of 500ms -> 120 BPM
    assert math.isclose(estimate_tempo_bpm([500, 500, 500]), 120.0)


def test_estimate_tempo_bpm_empty():
    assert estimate_tempo_bpm([]) is None


def test_bucket_tempo_boundaries():
    assert bucket_tempo(None) is None
    assert bucket_tempo(60) == "slow"
    assert bucket_tempo(110) == "medium"
    assert bucket_tempo(160) == "fast"
