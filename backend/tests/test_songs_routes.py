def test_search_songs_returns_results(client):
    resp = client.get("/songs/search", params={"q": "dancing queen"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["title"] == "Dancing Queen"
    assert body["results"][0]["artist"] == "ABBA"


def test_search_songs_no_results(client):
    resp = client.get("/songs/search", params={"q": "nothing matches this"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_search_songs_requires_query_param(client):
    resp = client.get("/songs/search")
    assert resp.status_code == 422


def test_lyrics_search_finds_song_by_lyrics_fragment(client):
    resp = client.get("/songs/lyrics-search", params={"q": "caught in a landslide"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["title"] == "Bohemian Rhapsody"
    assert "Is this the real life?" in results[0]["snippet"]


def test_lyrics_search_matches_trained_catalog_full_text(client):
    from tests.test_rhythm_routes import BASE_TAPS, train

    trained = train(client, "bohemian rhapsody", BASE_TAPS)

    # "just fantasy" appears only in the lyrics text, never in title/artist —
    # this must hit the local full-text path and identify our catalog song.
    resp = client.get("/songs/lyrics-search", params={"q": "just fantasy"})
    results = resp.json()["results"]

    assert results[0]["songId"] == trained["song"]["id"]
    assert "fantasy" in results[0]["snippet"].lower()


def test_training_stores_lyrics_on_song(client):
    from tests.test_rhythm_routes import BASE_TAPS, train

    train(client, "bohemian rhapsody", BASE_TAPS)
    # lyrics stored → local full-text search finds it with a catalog songId
    resp = client.get("/songs/lyrics-search", params={"q": "caught in a landslide"})
    assert resp.json()["results"][0]["songId"] is not None


def test_lyrics_search_no_results(client):
    resp = client.get("/songs/lyrics-search", params={"q": "zzz no such lyrics"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_get_lyrics_for_song(client):
    resp = client.get(
        "/songs/lyrics", params={"title": "Dancing Queen", "artist": "ABBA"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["lyrics"].startswith("You can dance")


def test_get_lyrics_not_found(client):
    resp = client.get("/songs/lyrics", params={"title": "Nope", "artist": "Nobody"})
    assert resp.status_code == 200
    assert resp.json()["lyrics"] is None


def test_preview_found(client):
    resp = client.get(
        "/songs/preview", params={"title": "Bohemian Rhapsody", "artist": "Queen"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["previewUrl"] == "https://example.com/preview.m4a"
    assert body["artworkUrl"] == "https://example.com/art600.jpg"


def test_preview_not_found(client):
    resp = client.get("/songs/preview", params={"title": "Unknown", "artist": "X"})
    assert resp.status_code == 200
    assert resp.json()["previewUrl"] is None


def _train_catalog(client):
    from tests.test_rhythm_routes import BASE_TAPS, train

    train(client, "bohemian rhapsody", BASE_TAPS)
    train(client, "dancing queen", [0, 150, 300, 450, 600, 750, 900])


def test_filter_by_artist_and_genre(client):
    _train_catalog(client)

    resp = client.get("/songs/filter", params={"artist": "queen"})
    names = [r["title"] for r in resp.json()["results"]]
    assert names == ["Bohemian Rhapsody"]

    resp = client.get("/songs/filter", params={"genre": "pop"})
    names = [r["title"] for r in resp.json()["results"]]
    assert names == ["Dancing Queen"]


def test_filter_by_decade(client):
    _train_catalog(client)
    resp = client.get("/songs/filter", params={"decade": "1970s"})
    names = {r["title"] for r in resp.json()["results"]}
    assert names == {"Bohemian Rhapsody", "Dancing Queen"}


def test_filter_no_params_returns_all(client):
    _train_catalog(client)
    resp = client.get("/songs/filter")
    assert len(resp.json()["results"]) == 2


def test_lyrics_search_degrades_gracefully_when_lrclib_times_out(client, fake_lyrics):
    import httpx

    def boom(query, limit=8):
        raise httpx.ReadTimeout("timed out")

    fake_lyrics.search = boom
    resp = client.get("/songs/lyrics-search", params={"q": "anything"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []

    resp = client.get("/songs/lyrics", params={"title": "A", "artist": "B"})
    assert resp.status_code == 200
    assert resp.json()["lyrics"] is None


def test_facets_reflect_catalog(client):
    _train_catalog(client)
    resp = client.get("/songs/facets")
    body = resp.json()
    assert body["genres"] == ["pop", "rock"]
    assert body["decades"] == ["1970s"]
    assert set(body["artists"]) == {"Queen", "ABBA"}
