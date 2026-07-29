"""30-second playback previews via the iTunes Search API — free, no key.
Spotify deprecated preview URLs for new API apps, so identified songs are
matched to iTunes by title+artist to obtain a playable preview clip."""

from dataclasses import dataclass
from typing import Protocol

import httpx

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


@dataclass
class PreviewHit:
    preview_url: str
    artwork_url: str | None
    itunes_track_name: str
    itunes_artist_name: str


class PreviewClientProtocol(Protocol):
    def find_preview(self, title: str, artist: str) -> PreviewHit | None: ...


class ITunesPreviewClient:
    def __init__(self, timeout: float = 8.0):
        self._timeout = timeout

    def find_preview(self, title: str, artist: str) -> PreviewHit | None:
        resp = httpx.get(
            ITUNES_SEARCH_URL,
            params={"term": f"{title} {artist}", "media": "music", "limit": 3},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        for item in resp.json().get("results", []):
            url = item.get("previewUrl")
            if url:
                artwork = item.get("artworkUrl100")
                if artwork:
                    # iTunes serves any square size by rewriting the suffix
                    artwork = artwork.replace("100x100", "600x600")
                return PreviewHit(
                    preview_url=url,
                    artwork_url=artwork,
                    itunes_track_name=item.get("trackName", ""),
                    itunes_artist_name=item.get("artistName", ""),
                )
        return None


_client: ITunesPreviewClient | None = None


def get_preview_client() -> PreviewClientProtocol:
    global _client
    if _client is None:
        _client = ITunesPreviewClient()
    return _client
