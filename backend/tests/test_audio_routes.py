from tests.audio_utils import MELODY_A, MELODY_B, clip, to_wav_bytes, tone_melody


def _upload(name: str = "clip.wav", samples=None):
    return {"file": (name, to_wav_bytes(samples), "audio/wav")}


def teach_audio(client, query: str, samples) -> dict:
    resp = client.post(
        "/audio/samples", data={"query": query}, files=_upload(samples=samples)
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_teach_audio_resolves_song_and_stores_fingerprints(client):
    body = teach_audio(client, "bohemian rhapsody", tone_melody(MELODY_A))
    assert body["status"] == "resolved"
    assert body["song"]["title"] == "Bohemian Rhapsody"
    assert body["song"]["audioSampleCount"] == 1


def test_audio_query_matches_taught_clip(client):
    taught = teach_audio(client, "bohemian rhapsody", tone_melody(MELODY_A))
    teach_audio(client, "dancing queen", tone_melody(MELODY_B))

    resp = client.post(
        "/audio/query", files=_upload(samples=clip(tone_melody(MELODY_A), 1.0, 5.5))
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["status"] == "confident"
    assert body["match"]["id"] == taught["song"]["id"]


def test_audio_query_unknown_audio_is_not_confident(client):
    teach_audio(client, "bohemian rhapsody", tone_melody(MELODY_A))

    resp = client.post("/audio/query", files=_upload(samples=tone_melody(MELODY_B)))
    assert resp.status_code == 200
    assert resp.json()["status"] != "confident"


def test_audio_query_empty_catalog_returns_no_match(client):
    resp = client.post("/audio/query", files=_upload(samples=tone_melody(MELODY_A)))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_match"
    assert body["candidates"] == []


def test_audio_query_rejects_too_short_clip(client):
    resp = client.post(
        "/audio/query", files=_upload(samples=clip(tone_melody(MELODY_A), 0.0, 0.5))
    )
    assert resp.status_code == 422


def test_teach_audio_unknown_song_not_found(client):
    resp = client.post(
        "/audio/samples",
        data={"query": "song that does not exist anywhere"},
        files=_upload(samples=tone_melody(MELODY_A)),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_found"


def test_audio_confirm_adds_fingerprints(client):
    taught = teach_audio(client, "bohemian rhapsody", tone_melody(MELODY_A))
    resp = client.post(
        "/audio/confirm",
        data={"song_id": taught["song"]["id"]},
        files=_upload(samples=clip(tone_melody(MELODY_A), 1.0, 5.5)),
    )
    assert resp.status_code == 200
    assert resp.json()["sampleCount"] == 2


def test_audio_confirm_404s_on_unknown_song(client):
    resp = client.post(
        "/audio/confirm",
        data={"song_id": "999999"},
        files=_upload(samples=tone_melody(MELODY_A)),
    )
    assert resp.status_code == 404


def test_audio_teach_seeds_auto_rhythm_sample(client):
    """Teaching by audio should make the song findable by TAPPING its rhythm,
    via the machine-extracted (quarantined) auto sample."""
    durations = [0.6, 0.3, 0.3, 0.6, 0.3, 0.6, 0.3, 0.3, 0.6, 0.6]
    roots = [440, 523, 659, 440, 587, 784, 659, 523, 440, 587]
    taught = teach_audio(client, "bohemian rhapsody", tone_melody(roots, durations))

    # a human taps the same long-short rhythm, 25% faster
    human_taps = [round(sum(durations[:i]) * 1000 / 1.25) for i in range(len(durations))]
    resp = client.post("/rhythm/query", json={"tapsMs": human_taps})
    assert resp.status_code == 200
    body = resp.json()

    ids = [c["id"] for c in body["candidates"]]
    assert taught["song"]["id"] in ids, "audio-taught song should surface as a rhythm candidate"


def test_rhythm_and_audio_share_one_catalog_entry(client):
    from tests.test_rhythm_routes import BASE_TAPS, train

    trained = train(client, "bohemian rhapsody", BASE_TAPS)
    taught = teach_audio(client, "bohemian rhapsody", tone_melody(MELODY_A))

    assert taught["song"]["id"] == trained["song"]["id"]
    assert taught["song"]["sampleCount"] == 1  # rhythm samples
    assert taught["song"]["audioSampleCount"] == 1  # audio clips
