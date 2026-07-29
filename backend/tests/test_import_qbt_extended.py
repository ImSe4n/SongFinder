import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_qbt_extended import parse_taps_ms  # noqa: E402


def test_parses_seconds_to_relative_ms():
    # 7.301535, 7.637346, 8.053503 -> ms, relative to first tap
    taps = parse_taps_ms("7.301535, 7.637346, 8.053503, 8.709644", min_taps=4, max_taps=100, max_span_ms=30_000)
    assert taps is not None
    assert taps[0] == 0
    assert taps == sorted(taps)
    assert all(taps[i] < taps[i + 1] for i in range(len(taps) - 1))


def test_rejects_too_few_taps():
    assert parse_taps_ms("1.0, 1.5, 2.0", min_taps=4, max_taps=100, max_span_ms=30_000) is None


def test_rejects_span_too_long():
    # 0s to 40s span, exceeds max_span_ms=30000
    taps = "0.0, 10.0, 20.0, 30.0, 40.0"
    assert parse_taps_ms(taps, min_taps=4, max_taps=100, max_span_ms=30_000) is None


def test_rejects_too_many_taps():
    seconds = ", ".join(str(i * 0.5) for i in range(150))
    assert parse_taps_ms(seconds, min_taps=4, max_taps=100, max_span_ms=30_000) is None


def test_deduplicates_taps_rounding_to_same_millisecond():
    # 1.0001 and 1.0002 both round to 1000ms -- must collapse to one point,
    # never violate the strictly-increasing requirement
    taps = parse_taps_ms(
        "0.0, 1.0001, 1.0002, 2.0, 3.0", min_taps=4, max_taps=100, max_span_ms=30_000
    )
    assert taps is not None
    assert all(taps[i] < taps[i + 1] for i in range(len(taps) - 1))


def test_handles_unsorted_input():
    # real data is already sorted, but don't assume it
    taps = parse_taps_ms("2.0, 0.0, 1.0, 3.0", min_taps=4, max_taps=100, max_span_ms=30_000)
    assert taps == [0, 1000, 2000, 3000]


def test_rejects_malformed_data():
    assert parse_taps_ms("not,a,number,here", min_taps=4, max_taps=100, max_span_ms=30_000) is None
    assert parse_taps_ms("", min_taps=4, max_taps=100, max_span_ms=30_000) is None


def test_real_dataset_row_parses_cleanly():
    real_row = (
        "7.301535, 7.637346, 8.053503, 8.709644, 9.668926, 10.980643, "
        "11.364697, 11.828319, 12.452086, 13.489938"
    )
    taps = parse_taps_ms(real_row, min_taps=4, max_taps=100, max_span_ms=30_000)
    assert taps is not None
    assert len(taps) == 10
    assert taps[0] == 0
    assert taps[-1] == round((13.489938 - 7.301535) * 1000)
