from collections import defaultdict

import numpy as np

from app.audio.decode import decode_to_mono
from app.audio.fingerprint import fingerprint
from app.audio.matching import decide_audio_match, rank_audio_matches
from tests.audio_utils import MELODY_A, MELODY_B, clip, to_wav_bytes, tone_melody


def _db_hits_for(song_hashes: dict[int, list[tuple[int, int]]], query):
    """Simulate the DB lookup: map each query hash to stored (song_id, offset)."""
    index = defaultdict(list)
    for song_id, hashes in song_hashes.items():
        for h, t in hashes:
            index[h].append((song_id, t))
    return {h: index[h] for h, _ in query if h in index}


def test_fingerprint_is_deterministic_and_nonempty():
    samples = tone_melody(MELODY_A)
    fp1 = fingerprint(samples)
    fp2 = fingerprint(samples)
    assert fp1 == fp2
    assert len(fp1) > 50


def test_clip_matches_full_recording():
    full_a = fingerprint(tone_melody(MELODY_A))
    full_b = fingerprint(tone_melody(MELODY_B))
    query = fingerprint(clip(tone_melody(MELODY_A), 1.0, 5.5))

    ranked = rank_audio_matches(query, _db_hits_for({1: full_a, 2: full_b}, query))
    status, top = decide_audio_match(ranked)

    assert top is not None
    assert top.song_id == 1
    assert status == "confident"


def test_unrelated_audio_does_not_match():
    full_a = fingerprint(tone_melody(MELODY_A))
    query = fingerprint(tone_melody(MELODY_B))

    ranked = rank_audio_matches(query, _db_hits_for({1: full_a}, query))
    status, _ = decide_audio_match(ranked)

    assert status != "confident"


def test_wav_roundtrip_through_decoder():
    samples = tone_melody(MELODY_A)
    decoded = decode_to_mono(to_wav_bytes(samples))
    # decode(encode(x)) won't be bit-identical (PCM16 quantization), but the
    # fingerprints should still align to the same song at offset ~0
    fp_original = fingerprint(samples)
    fp_decoded = fingerprint(decoded)

    ranked = rank_audio_matches(fp_decoded, _db_hits_for({1: fp_original}, fp_decoded))
    status, top = decide_audio_match(ranked)
    assert status == "confident"
    assert top.song_id == 1


def test_too_short_input_yields_no_hashes():
    assert fingerprint(np.zeros(100, dtype=np.float32)) == []
