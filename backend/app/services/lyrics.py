"""Lyrics via lrclib.net — a free, open lyrics database with a public API
(no key or signup required). Used for both lyrics-fragment search ("I only
remember the words") and fetching full lyrics for an identified song."""

from dataclasses import dataclass
from typing import Protocol

import httpx

LRCLIB_BASE = "https://lrclib.net/api"


@dataclass
class LyricsHit:
    title: str
    artist: str
    album: str | None
    plain_lyrics: str | None


class LyricsClientProtocol(Protocol):
    def search(self, query: str, limit: int = 8) -> list[LyricsHit]: ...

    def get_lyrics(self, title: str, artist: str) -> LyricsHit | None: ...


class LrclibClient:
    def __init__(self, timeout: float = 8.0):
        self._timeout = timeout

    def _to_hits(self, payload: list[dict]) -> list[LyricsHit]:
        return [
            LyricsHit(
                title=item.get("trackName") or "",
                artist=item.get("artistName") or "",
                album=item.get("albumName"),
                plain_lyrics=item.get("plainLyrics"),
            )
            for item in payload
            if item.get("trackName")
        ]

    def search(self, query: str, limit: int = 8) -> list[LyricsHit]:
        resp = httpx.get(
            f"{LRCLIB_BASE}/search", params={"q": query}, timeout=self._timeout
        )
        resp.raise_for_status()
        return self._to_hits(resp.json())[:limit]

    def get_lyrics(self, title: str, artist: str) -> LyricsHit | None:
        # lrclib's /get needs exact fields; its /search is far more forgiving,
        # so search and take the best hit that actually has lyrics.
        hits = self.search(f"{title} {artist}", limit=8)
        with_lyrics = [h for h in hits if h.plain_lyrics]
        return with_lyrics[0] if with_lyrics else (hits[0] if hits else None)


_client: LrclibClient | None = None


def get_lyrics_client() -> LyricsClientProtocol:
    global _client
    if _client is None:
        _client = LrclibClient()
    return _client
