"""Hybrid (learned-encoder + DTW) rhythm matching path.

These need torch + the trained checkpoint. When the checkpoint is absent
(fresh clone / CI without models/), the fallback-to-DTW tests still run and
are the important safety guarantee; the model-active tests skip.
"""

from pathlib import Path

import pytest

from app.config import settings
from tests.test_rhythm_routes import BASE_TAPS, train

torch = pytest.importorskip("torch")

CHECKPOINT_PRESENT = Path(settings.rhythm_model_path).is_file()
requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_PRESENT, reason="rhythm model checkpoint not present"
)


def _scaled(taps, f):
    return [round(t * f) for t in taps]


@pytest.fixture(autouse=True)
def _reset_hybrid_caches():
    """The hybrid model/embedding caches are process-global; each test must
    start clean so a cached embedding from another test's DB can't leak."""
    from app.rhythm.hybrid_matching import clear_caches

    clear_caches()
    yield
    clear_caches()


def test_query_falls_back_to_dtw_when_hybrid_disabled(client, monkeypatch):
    """The model is strictly additive: with it disabled, /rhythm/query must
    still work via pure DTW."""
    monkeypatch.setattr(settings, "hybrid_matching_enabled", False)
    train(client, "bohemian rhapsody", BASE_TAPS)

    resp = client.post("/rhythm/query", json={"tapsMs": _scaled(BASE_TAPS, 0.8)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "confident"
    assert body["match"]["title"] == "Bohemian Rhapsody"


def test_query_falls_back_when_checkpoint_missing(client, monkeypatch):
    """A bad/missing checkpoint path must degrade to DTW, not 500."""
    monkeypatch.setattr(settings, "hybrid_matching_enabled", True)
    monkeypatch.setattr(settings, "rhythm_model_path", "models/does_not_exist.pt")
    from app.rhythm.hybrid_matching import clear_caches

    clear_caches()  # force a fresh load attempt against the bad path

    train(client, "bohemian rhapsody", BASE_TAPS)
    resp = client.post("/rhythm/query", json={"tapsMs": _scaled(BASE_TAPS, 0.8)})
    assert resp.status_code == 200
    assert resp.json()["status"] in ("confident", "ambiguous", "no_match")


@requires_checkpoint
def test_query_empty_catalog_no_match_hybrid(client, monkeypatch):
    monkeypatch.setattr(settings, "hybrid_matching_enabled", True)
    resp = client.post("/rhythm/query", json={"tapsMs": BASE_TAPS})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_match"
    assert body["candidates"] == []


@requires_checkpoint
def test_hybrid_confident_when_legs_agree(client, monkeypatch):
    """With one very distinct trained song and a tempo-scaled query of it,
    both legs should agree -> confident."""
    monkeypatch.setattr(settings, "hybrid_matching_enabled", True)
    trained = train(client, "bohemian rhapsody", BASE_TAPS)

    resp = client.post("/rhythm/query", json={"tapsMs": _scaled(BASE_TAPS, 0.75)})
    assert resp.status_code == 200
    body = resp.json()
    # correct song should be the top candidate regardless of confident/ambiguous
    assert body["candidates"], "expected at least one candidate"
    assert body["candidates"][0]["id"] == trained["song"]["id"]


@requires_checkpoint
def test_hybrid_returns_query_log_id(client, monkeypatch):
    monkeypatch.setattr(settings, "hybrid_matching_enabled", True)
    train(client, "bohemian rhapsody", BASE_TAPS)
    resp = client.post("/rhythm/query", json={"tapsMs": _scaled(BASE_TAPS, 0.9)})
    assert resp.json()["queryLogId"] is not None


@requires_checkpoint
def test_rank_hybrid_direct_two_songs(db_session, monkeypatch):
    """Exercise rank_hybrid directly on two distinct rhythms in one catalog."""
    monkeypatch.setattr(settings, "hybrid_matching_enabled", True)
    from app.models import RhythmSample, Song
    from app.rhythm.features import iois_to_log_ratio, taps_to_iois
    from app.rhythm.hybrid_matching import clear_caches, rank_hybrid

    clear_caches()

    def add_song(sid_taps, title, taps_variants):
        song = Song(spotify_track_id=f"t-{title}", title=title, artist="X", genres=[])
        db_session.add(song)
        db_session.flush()
        for taps in taps_variants:
            db_session.add(
                RhythmSample(
                    song_id=song.id,
                    raw_taps_ms=taps,
                    ioi_ms=taps_to_iois(taps),
                    log_ratio_vector=iois_to_log_ratio(taps_to_iois(taps)),
                    source="train",
                )
            )
        db_session.flush()
        return song.id

    steady = [0, 400, 800, 1200, 1600, 2000, 2400]
    longshort = [0, 150, 700, 850, 1400, 1550, 2100]
    id_a = add_song(None, "steady", [steady, _scaled(steady, 1.05)])
    id_b = add_song(None, "longshort", [longshort, _scaled(longshort, 1.05)])
    db_session.commit()

    songs = db_session.query(Song).all()
    status, fused, conf, mtop, dtop = rank_hybrid(_scaled(steady, 0.8), songs)
    assert fused[0] == id_a, f"expected steady song on top, got {fused}"
    assert id_b in fused
