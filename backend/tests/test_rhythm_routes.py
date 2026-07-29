BASE_TAPS = [0, 200, 500, 900, 1200, 1600]


def _scaled(taps: list[int], factor: float) -> list[int]:
    return [round(t * factor) for t in taps]


def train(client, query: str, taps: list[int]) -> dict:
    resp = client.post("/rhythm/samples", json={"query": query, "tapsMs": taps})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_training_resolves_via_fake_spotify_and_creates_song(client, fake_spotify):
    body = train(client, "bohemian rhapsody", BASE_TAPS)

    assert body["status"] == "resolved"
    assert body["song"]["title"] == "Bohemian Rhapsody"
    assert body["song"]["artist"] == "Queen"
    assert body["song"]["sampleCount"] == 1
    assert fake_spotify.search_calls == 1


def test_second_training_submission_uses_local_catalog_first(client, fake_spotify):
    from app.services.spotify import ResolvedTrack

    fake_spotify.tracks = {
        "clean bandit": ResolvedTrack(
            spotify_track_id="wrong-track",
            title="Another Song",
            artist="Clean Bandit",
            genres=["pop"],
            release_year=2014,
            album_art_url="https://example.com/wrong.jpg",
        ),
        "rather be": ResolvedTrack(
            spotify_track_id="track-rb",
            title="Rather Be",
            artist="Clean Bandit",
            genres=["pop"],
            release_year=2014,
            album_art_url="https://example.com/rb.jpg",
        ),
    }

    train(client, "Rather Be", BASE_TAPS)
    calls_after_first = fake_spotify.search_calls

    body = train(client, "Clean Bandit - Rather Be", _scaled(BASE_TAPS, 1.1))

    assert body["status"] == "resolved"
    assert body["song"]["sampleCount"] == 2
    assert fake_spotify.search_calls == calls_after_first


def test_training_not_found_for_unknown_song(client):
    resp = client.post(
        "/rhythm/samples", json={"query": "some unknown song nobody made", "tapsMs": BASE_TAPS}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_found"
    assert body["song"] is None


def test_query_matches_tempo_scaled_trained_rhythm(client):
    trained = train(client, "bohemian rhapsody", BASE_TAPS)
    song_id = trained["song"]["id"]

    resp = client.post("/rhythm/query", json={"tapsMs": _scaled(BASE_TAPS, 0.6)})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["status"] == "confident"
    assert body["match"]["id"] == song_id
    assert len(body["candidates"]) >= 1


def test_query_unrelated_rhythm_not_confidently_matched(client):
    train(client, "bohemian rhapsody", BASE_TAPS)

    unrelated = [0, 90, 110, 130, 2000, 2500, 2600]
    resp = client.post("/rhythm/query", json={"tapsMs": unrelated})
    body = resp.json()

    assert body["status"] != "confident"


def test_query_on_empty_catalog_returns_no_match(client):
    resp = client.post("/rhythm/query", json={"tapsMs": BASE_TAPS})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_match"
    assert body["match"] is None
    assert body["candidates"] == []


def test_confirm_increments_sample_count(client):
    trained = train(client, "bohemian rhapsody", BASE_TAPS)
    song_id = trained["song"]["id"]

    resp = client.post(
        "/rhythm/confirm", json={"songId": song_id, "tapsMs": _scaled(BASE_TAPS, 1.2)}
    )
    assert resp.status_code == 200
    assert resp.json()["sampleCount"] == 2


def test_confirm_404s_on_unknown_song(client):
    resp = client.post("/rhythm/confirm", json={"songId": "999999", "tapsMs": BASE_TAPS})
    assert resp.status_code == 404


def test_query_rejects_too_short_tap_sequence(client):
    resp = client.post("/rhythm/query", json={"tapsMs": [0, 200]})
    assert resp.status_code == 422


def test_query_rejects_non_increasing_taps(client):
    resp = client.post("/rhythm/query", json={"tapsMs": [0, 200, 150, 400, 600]})
    assert resp.status_code == 422


def test_query_rejects_taps_not_starting_at_zero(client):
    resp = client.post("/rhythm/query", json={"tapsMs": [10, 200, 400, 600, 800]})
    assert resp.status_code == 422


def test_disambiguates_two_distinct_trained_songs(client):
    # a starkly different long-short-long-short pattern, not just a jittered
    # variant of BASE_TAPS, so the two songs are clearly separable
    dancing_taps = [0, 100, 700, 800, 1400, 1500, 2100]
    trained_a = train(client, "bohemian rhapsody", BASE_TAPS)
    trained_b = train(client, "dancing queen", dancing_taps)

    resp_a = client.post("/rhythm/query", json={"tapsMs": _scaled(BASE_TAPS, 0.8)})
    resp_b = client.post("/rhythm/query", json={"tapsMs": _scaled(dancing_taps, 1.3)})

    assert resp_a.json()["match"]["id"] == trained_a["song"]["id"]
    assert resp_b.json()["match"]["id"] == trained_b["song"]["id"]
