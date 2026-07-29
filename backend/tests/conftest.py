import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db, get_session_factory
from app.main import app
from app.services.lyrics import LyricsHit, get_lyrics_client
from app.services.previews import PreviewHit, get_preview_client
from app.services.spotify import ResolvedTrack, get_spotify_client


class FakeSpotifyClient:
    """Canned Spotify responses, keyed by a lowercase substring of the query.
    Tracks call counts so tests can assert local-catalog-first behavior."""

    def __init__(self, tracks: dict[str, ResolvedTrack] | None = None):
        self.tracks = tracks or {}
        self.search_calls = 0
        self.genre_calls = 0

    def _match(self, query: str) -> ResolvedTrack | None:
        needle = query.lower()
        for key, resolved in self.tracks.items():
            if key in needle or needle in key:
                return resolved
        return None

    def search_track(self, query: str, limit: int = 5) -> list[dict]:
        self.search_calls += 1
        resolved = self._match(query)
        if resolved is None:
            return []
        return [
            {
                "id": resolved.spotify_track_id,
                "name": resolved.title,
                "artists": [
                    {"id": f"artist-{resolved.spotify_track_id}", "name": resolved.artist}
                ],
                "album": {
                    "images": (
                        [{"url": resolved.album_art_url}] if resolved.album_art_url else []
                    ),
                    "release_date": (
                        f"{resolved.release_year}-01-01" if resolved.release_year else None
                    ),
                },
            }
        ]

    def get_artist_genres(self, artist_id: str) -> list[str]:
        self.genre_calls += 1
        for resolved in self.tracks.values():
            if f"artist-{resolved.spotify_track_id}" == artist_id:
                return resolved.genres
        return []

    def resolve_track(self, query: str) -> ResolvedTrack | None:
        # Mirrors the real SpotifyClient.resolve_track: goes through
        # search_track (so call-counting in tests reflects real behavior)
        # rather than matching directly.
        results = self.search_track(query, limit=1)
        if not results:
            return None
        return self._match(query)


@pytest.fixture()
def fake_spotify() -> FakeSpotifyClient:
    return FakeSpotifyClient(
        {
            "bohemian rhapsody": ResolvedTrack(
                spotify_track_id="track1",
                title="Bohemian Rhapsody",
                artist="Queen",
                genres=["rock"],
                release_year=1975,
                album_art_url="https://example.com/art1.jpg",
            ),
            "dancing queen": ResolvedTrack(
                spotify_track_id="track2",
                title="Dancing Queen",
                artist="ABBA",
                genres=["pop"],
                release_year=1976,
                album_art_url="https://example.com/art2.jpg",
            ),
        }
    )


class FakeLyricsClient:
    def __init__(self):
        self.hits = [
            LyricsHit(
                title="Bohemian Rhapsody",
                artist="Queen",
                album="A Night at the Opera",
                plain_lyrics="Is this the real life?\nIs this just fantasy?\nCaught in a landslide",
            ),
            LyricsHit(
                title="Dancing Queen",
                artist="ABBA",
                album="Arrival",
                plain_lyrics="You can dance\nYou can jive\nHaving the time of your life",
            ),
        ]

    def search(self, query: str, limit: int = 8):
        needle = query.lower()
        return [
            h
            for h in self.hits
            if needle in f"{h.title} {h.artist} {h.plain_lyrics or ''}".lower()
        ][:limit]

    def get_lyrics(self, title: str, artist: str):
        for h in self.hits:
            if h.title.lower() == title.lower():
                return h
        return None


class FakePreviewClient:
    def find_preview(self, title: str, artist: str):
        if "bohemian" in title.lower():
            return PreviewHit(
                preview_url="https://example.com/preview.m4a",
                artwork_url="https://example.com/art600.jpg",
                itunes_track_name=title,
                itunes_artist_name=artist,
            )
        return None


@pytest.fixture()
def fake_lyrics() -> FakeLyricsClient:
    return FakeLyricsClient()


@pytest.fixture()
def fake_previews() -> FakePreviewClient:
    return FakePreviewClient()


@pytest.fixture()
def db_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture()
def client(
    db_engine, fake_spotify: FakeSpotifyClient, fake_lyrics: FakeLyricsClient, fake_previews: FakePreviewClient
):
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    # Background tasks (e.g. auto-rhythm extraction) open their OWN session
    # via Depends(get_session_factory) rather than Depends(get_db) — must be
    # overridden separately or they'd silently hit the real configured DB.
    app.dependency_overrides[get_session_factory] = lambda: TestingSessionLocal
    app.dependency_overrides[get_spotify_client] = lambda: fake_spotify
    app.dependency_overrides[get_lyrics_client] = lambda: fake_lyrics
    app.dependency_overrides[get_preview_client] = lambda: fake_previews

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture()
def db_session(db_engine):
    """Direct access to the SAME test database the `client` fixture writes
    to, for assertions that need to inspect persisted rows rather than just
    HTTP response bodies. Must be used together with `client` in the same
    test (order of fixture args doesn't matter — both resolve to db_engine)."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
