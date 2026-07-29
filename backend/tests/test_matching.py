import math

from app.rhythm.features import iois_to_log_ratio, taps_to_iois
from app.rhythm.matching import (
    confidence_from_distance,
    decide_match,
    dtw_distance,
    rank_candidates,
    subsequence_dtw_distance,
)


def _r(taps_ms: list[int]) -> list[float]:
    return iois_to_log_ratio(taps_to_iois(taps_ms))


def test_identical_rhythm_same_tempo_has_near_zero_distance():
    taps = [0, 200, 500, 900, 1200]
    d = dtw_distance(_r(taps), _r(taps))
    assert d < 1e-6


def test_identical_rhythm_different_tempo_still_near_zero():
    """The single most important test: tempo invariance. The same rhythm
    tapped twice as fast must still match closely."""
    taps = [0, 200, 500, 900, 1200]
    taps_fast = [round(t / 2) for t in taps]
    d = dtw_distance(_r(taps), _r(taps_fast))
    assert d < 0.05


def test_different_rhythms_have_large_distance():
    steady = [0, 200, 400, 600, 800, 1000]
    syncopated = [0, 150, 700, 850, 900, 1400]
    d = dtw_distance(_r(steady), _r(syncopated))
    unrelated_confidence = confidence_from_distance(d)
    same_confidence = confidence_from_distance(dtw_distance(_r(steady), _r(steady)))
    assert unrelated_confidence < same_confidence


def test_jittered_variant_is_closer_than_unrelated_rhythm():
    base = [0, 200, 500, 900, 1200, 1600]
    jittered = [0, 220, 480, 950, 1150, 1650]
    unrelated = [0, 300, 320, 340, 1500, 1520]

    d_jittered = dtw_distance(_r(base), _r(jittered))
    d_unrelated = dtw_distance(_r(base), _r(unrelated))
    assert d_jittered < d_unrelated


def test_extra_tap_still_reasonably_close():
    base = [0, 200, 500, 900, 1200, 1600]
    # an extra tap inserted in the middle
    with_extra = [0, 200, 350, 500, 900, 1200, 1600]
    d = dtw_distance(_r(base), _r(with_extra))
    assert d < 1.0


def test_rank_candidates_orders_by_best_matching_sample():
    query_taps = [0, 200, 500, 900, 1200]
    query_r = _r(query_taps)

    matching_song_samples = [_r(query_taps), _r([0, 210, 490, 910, 1190])]
    unrelated_song_samples = [_r([0, 300, 320, 340, 1500])]

    ranked = rank_candidates(
        query_r,
        [
            (1, unrelated_song_samples, []),
            (2, matching_song_samples, []),
        ],
    )
    assert ranked[0].song_id == 2
    assert ranked[0].confidence > ranked[1].confidence


def test_decide_match_confident_when_clear_winner():
    ranked = rank_candidates(
        _r([0, 200, 500, 900, 1200]),
        [
            (1, [_r([0, 200, 500, 900, 1200])], []),
            (2, [_r([0, 300, 320, 340, 1500])], []),
        ],
    )
    status, top = decide_match(ranked, threshold=70, margin=15)
    assert status == "confident"
    assert top.song_id == 1


def test_decide_match_ambiguous_when_close_scores():
    ranked = rank_candidates(
        _r([0, 200, 500, 900, 1200]),
        [
            (1, [_r([0, 200, 500, 900, 1200])], []),
            (2, [_r([0, 205, 505, 895, 1205])], []),
        ],
    )
    status, top = decide_match(ranked, threshold=70, margin=15)
    assert status == "ambiguous"
    assert top is not None


def test_auto_samples_are_penalized_vs_human_samples():
    query = _r([0, 200, 500, 900, 1200])

    # Same perfect-matching vector: song 1 has it as a human sample, song 2
    # only as a machine-extracted 'auto' sample.
    ranked = rank_candidates(
        query,
        [
            (1, [_r([0, 200, 500, 900, 1200])], []),
            (2, [], [_r([0, 200, 500, 900, 1200])]),
        ],
    )
    human = next(r for r in ranked if r.song_id == 1)
    auto = next(r for r in ranked if r.song_id == 2)
    assert human.confidence > auto.confidence
    assert ranked[0].song_id == 1


def test_decide_match_no_match_on_empty_catalog():
    status, top = decide_match([], threshold=70, margin=15)
    assert status == "no_match"
    assert top is None


LONG_REFERENCE = [0.4, -0.2, 0.7, 0.1, -0.5, 0.3, 0.6, -0.1, 0.2, -0.4, 0.5, 0.0, -0.3, 0.4, 0.1, -0.6, 0.2, 0.3]


def test_subsequence_dtw_finds_exact_fragment_at_zero_distance():
    fragment = LONG_REFERENCE[6:12]  # a contiguous slice from the middle
    assert subsequence_dtw_distance(fragment, LONG_REFERENCE) == 0.0


def test_subsequence_dtw_fixes_global_dtw_short_query_failure():
    """The motivating case: an exact fragment of a long reference. Global
    DTW must explain the reference's unmatched head and tail so its distance
    is large; subsequence DTW charges only the aligned region."""
    fragment = LONG_REFERENCE[6:12]
    assert subsequence_dtw_distance(fragment, LONG_REFERENCE) == 0.0
    assert dtw_distance(fragment, LONG_REFERENCE) > 0.05


def test_subsequence_dtw_unrelated_fragment_scores_worse():
    fragment = LONG_REFERENCE[6:12]
    unrelated = [2.0, -2.0, 2.0, -2.0, 2.0, -2.0]
    assert subsequence_dtw_distance(unrelated, LONG_REFERENCE) > subsequence_dtw_distance(
        fragment, LONG_REFERENCE
    )


def test_subsequence_dtw_argument_order_is_symmetric():
    fragment = LONG_REFERENCE[6:12]
    assert subsequence_dtw_distance(fragment, LONG_REFERENCE) == subsequence_dtw_distance(
        LONG_REFERENCE, fragment
    )


def test_subsequence_dtw_equal_length_identical_is_zero():
    assert subsequence_dtw_distance(LONG_REFERENCE, LONG_REFERENCE) == 0.0


def test_subsequence_dtw_empty_handling():
    assert subsequence_dtw_distance([], []) == 0.0
    assert subsequence_dtw_distance([], [0.1, 0.2]) > 0.0


def test_confidence_from_distance_is_clamped():
    assert confidence_from_distance(0.0) == 100.0
    assert confidence_from_distance(1000.0) == 0.0 or confidence_from_distance(1000.0) >= 0.0
    assert math.isfinite(confidence_from_distance(1000.0))
