import threading
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import settings
from app.services.query_matching import normalize_query, query_variants, structured_query_parts

TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"
TOKEN_SAFETY_BUFFER_SECONDS = 30


@dataclass
class ResolvedTrack:
    spotify_track_id: str
    title: str
    artist: str
    genres: list[str]
    release_year: int | None
    album_art_url: str | None


class SpotifyClientProtocol(Protocol):
    def search_track(self, query: str, limit: int = 5) -> list[dict]: ...

    def get_artist_genres(self, artist_id: str) -> list[str]: ...

    def resolve_track(self, query: str) -> ResolvedTrack | None: ...


def _parse_release_year(release_date: str | None) -> int | None:
    if not release_date:
        return None
    try:
        return int(release_date[:4])
    except (ValueError, TypeError):
        return None


def _track_to_resolved(track: dict, genres: list[str]) -> ResolvedTrack:
    artists = track.get("artists") or []
    artist_name = artists[0]["name"] if artists else "Unknown"
    images = (track.get("album") or {}).get("images") or []
    album_art_url = images[0]["url"] if images else None
    return ResolvedTrack(
        spotify_track_id=track["id"],
        title=track["name"],
        artist=artist_name,
        genres=genres,
        release_year=_parse_release_year((track.get("album") or {}).get("release_date")),
        album_art_url=album_art_url,
    )


def _track_score(track: dict, query: str) -> int:
    title = normalize_query(track.get("name") or "")
    artists = track.get("artists") or []
    artist = normalize_query(artists[0]["name"] if artists else "Unknown")
    combined = f"{title} {artist}".strip()
    if not combined:
        return 0

    score = 0
    parts = structured_query_parts(query)
    if parts is not None:
        first, second = parts
        if first in artist and second in title:
            score = max(score, 4)
        if first in title and second in artist:
            score = max(score, 4)

    for needle in query_variants(query):
        if needle == combined or needle == f"{artist} {title}":
            return 6
        if needle == title or needle == artist:
            score = max(score, 3)
        elif needle in combined or combined in needle:
            score = max(score, 1)

    return score


class SpotifyClient:
    """Client Credentials flow only (server-to-server). No redirect URI is
    needed for this — that's only required for user-login/Authorization Code
    flows, which this app does not use."""

    def __init__(self, client_id: str, client_secret: str, timeout: float = 8.0):
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout
        self._lock = threading.Lock()
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def _get_token(self) -> str:
        with self._lock:
            if self._token and time.monotonic() < self._token_expires_at:
                return self._token
            resp = httpx.post(
                TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(self._client_id, self._client_secret),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload["access_token"]
            expires_in = payload.get("expires_in", 3600)
            self._token_expires_at = time.monotonic() + expires_in - TOKEN_SAFETY_BUFFER_SECONDS
            return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}"}

    def search_track(self, query: str, limit: int = 5) -> list[dict]:
        resp = httpx.get(
            f"{API_BASE}/search",
            params={"q": query, "type": "track", "limit": limit},
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("tracks", {}).get("items", [])

    def get_artist_genres(self, artist_id: str) -> list[str]:
        resp = httpx.get(
            f"{API_BASE}/artists/{artist_id}",
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("genres", [])

    def resolve_track(self, query: str) -> ResolvedTrack | None:
        results_by_id: dict[str, dict] = {}
        for variant in query_variants(query):
            results = self.search_track(variant, limit=5)
            for track in results:
                track_id = track.get("id")
                if track_id and track_id not in results_by_id:
                    results_by_id[track_id] = track

        if not results_by_id:
            return None

        track = max(results_by_id.values(), key=lambda candidate: _track_score(candidate, query))
        artists = track.get("artists") or []
        genres = self.get_artist_genres(artists[0]["id"]) if artists else []
        return _track_to_resolved(track, genres)


_client: SpotifyClient | None = None


def get_spotify_client() -> SpotifyClientProtocol:
    global _client
    if _client is None:
        _client = SpotifyClient(settings.spotify_client_id, settings.spotify_client_secret)
    return _client
