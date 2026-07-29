from tests.audio_utils import MELODY_A, to_wav_bytes, tone_melody
from tests.test_rhythm_routes import BASE_TAPS, train


def _scaled(taps: list[int], factor: float) -> list[int]:
    return [round(t * factor) for t in taps]


def test_rhythm_query_returns_a_log_id(client):
    train(client, "bohemian rhapsody", BASE_TAPS)
    resp = client.post("/rhythm/query", json={"tapsMs": _scaled(BASE_TAPS, 0.8)})
    body = resp.json()
    assert body["queryLogId"] is not None
    assert body["status"] == "confident"


def test_rhythm_query_persists_taps_and_result(client, db_session):
    from app.models import QueryLog

    trained = train(client, "bohemian rhapsody", BASE_TAPS)
    query_taps = _scaled(BASE_TAPS, 0.8)
    resp = client.post("/rhythm/query", json={"tapsMs": query_taps})
    log_id = int(resp.json()["queryLogId"])

    entry = db_session.get(QueryLog, log_id)
    assert entry is not None
    assert entry.query_type == "rhythm"
    assert entry.status == "confident"
    assert entry.taps_ms == query_taps
    assert entry.top_song_id == int(trained["song"]["id"])
    assert entry.outcome is None  # no feedback submitted yet


def test_feedback_saved_round_trip(client, db_session):
    from app.models import QueryLog

    trained = train(client, "bohemian rhapsody", BASE_TAPS)
    resp = client.post("/rhythm/query", json={"tapsMs": _scaled(BASE_TAPS, 0.8)})
    body = resp.json()
    log_id = body["queryLogId"]
    song_id = body["match"]["id"]

    fb = client.post(
        "/rhythm/feedback",
        json={"queryLogId": log_id, "outcome": "saved", "songId": song_id},
    )
    assert fb.status_code == 200
    assert fb.json()["status"] == "ok"

    entry = db_session.get(QueryLog, int(log_id))
    assert entry.outcome == "saved"
    assert entry.outcome_song_id == int(song_id)
    assert entry.responded_at is not None


def test_feedback_not_it(client):
    train(client, "bohemian rhapsody", BASE_TAPS)
    resp = client.post("/rhythm/query", json={"tapsMs": _scaled(BASE_TAPS, 0.8)})
    log_id = resp.json()["queryLogId"]

    fb = client.post("/rhythm/feedback", json={"queryLogId": log_id, "outcome": "not_it"})
    assert fb.status_code == 200


def test_feedback_saved_without_song_id_is_rejected(client):
    train(client, "bohemian rhapsody", BASE_TAPS)
    resp = client.post("/rhythm/query", json={"tapsMs": _scaled(BASE_TAPS, 0.8)})
    log_id = resp.json()["queryLogId"]

    fb = client.post("/rhythm/feedback", json={"queryLogId": log_id, "outcome": "saved"})
    assert fb.status_code == 422


def test_feedback_unknown_log_id_404s(client):
    fb = client.post("/rhythm/feedback", json={"queryLogId": "999999", "outcome": "not_it"})
    assert fb.status_code == 404


def test_feedback_malformed_log_id_404s(client):
    fb = client.post("/rhythm/feedback", json={"queryLogId": "not-a-number", "outcome": "not_it"})
    assert fb.status_code == 404


def test_audio_query_persists_hash_count_no_audio_stored(client, db_session):
    from app.models import QueryLog

    resp = client.post(
        "/audio/query",
        files={"file": ("clip.wav", to_wav_bytes(tone_melody(MELODY_A)), "audio/wav")},
    )
    assert resp.status_code == 200
    log_id = int(resp.json()["queryLogId"])

    entry = db_session.get(QueryLog, log_id)
    assert entry is not None
    assert entry.query_type == "audio"
    assert entry.audio_hash_count is not None and entry.audio_hash_count > 0
    assert entry.taps_ms is None  # audio queries never store rhythm data
