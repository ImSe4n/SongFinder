from app.services.spotify import SpotifyClient


def test_resolve_track_prefers_structured_artist_title_match():
    client = SpotifyClient("client-id", "client-secret")

    wrong = {
        "id": "wrong-track",
        "name": "Another Song",
        "artists": [{"id": "wrong-artist", "name": "Clean Bandit"}],
        "album": {"images": [], "release_date": "2014-01-01"},
    }
    correct = {
        "id": "right-track",
        "name": "Rather Be",
        "artists": [{"id": "right-artist", "name": "Clean Bandit"}],
        "album": {"images": [], "release_date": "2014-01-01"},
    }

    def fake_search_track(query: str, limit: int = 5):
        return [wrong, correct]

    def fake_get_artist_genres(artist_id: str):
        return ["pop"]

    client.search_track = fake_search_track  # type: ignore[method-assign]
    client.get_artist_genres = fake_get_artist_genres  # type: ignore[method-assign]

    resolved = client.resolve_track("Clean Bandit - Rather Be")

    assert resolved is not None
    assert resolved.spotify_track_id == "right-track"
    assert resolved.title == "Rather Be"
    assert resolved.artist == "Clean Bandit"