import httpx
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Song
from app.schemas.song import (
    FacetsResponse,
    FilterResponse,
    LyricsResponse,
    LyricsSearchResponse,
    LyricsSearchResult,
    PreviewResponse,
    SongOut,
    SongSearchResponse,
    SongSearchResult,
)
from app.services.lyrics import LyricsClientProtocol, get_lyrics_client
from app.services.previews import PreviewClientProtocol, get_preview_client
from app.services.spotify import SpotifyClientProtocol, get_spotify_client

router = APIRouter(prefix="/songs", tags=["songs"])

SNIPPET_LINES = 3


@router.get("/search", response_model=SongSearchResponse)
def search_songs(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    spotify_client: SpotifyClientProtocol = Depends(get_spotify_client),
) -> SongSearchResponse:
    try:
        tracks = spotify_client.search_track(q, limit=limit)
    except httpx.HTTPError:
        # External service being slow/down should degrade to empty results,
        # not a 500.
        return SongSearchResponse(results=[])
    results = [
        SongSearchResult(
            spotify_track_id=track["id"],
            title=track["name"],
            artist=(track.get("artists") or [{"name": "Unknown"}])[0]["name"],
            album_art_url=(((track.get("album") or {}).get("images") or [{}])[0]).get("url"),
        )
        for track in tracks
    ]
    return SongSearchResponse(results=results)


def _snippet(lyrics: str | None) -> str | None:
    if not lyrics:
        return None
    lines = [ln.strip() for ln in lyrics.splitlines() if ln.strip()]
    return " / ".join(lines[:SNIPPET_LINES]) or None


def _snippet_around(lyrics: str, needle: str) -> str | None:
    """The matching line plus the next one — shows *why* this song matched."""
    lines = [ln.strip() for ln in lyrics.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if needle in line.lower():
            return " / ".join(lines[i : i + 2])
    return " / ".join(lines[:SNIPPET_LINES]) or None


@router.get("/lyrics-search", response_model=LyricsSearchResponse)
def lyrics_search(
    q: str = Query(..., min_length=2),
    limit: int = Query(8, ge=1, le=20),
    db: Session = Depends(get_db),
    lyrics_client: LyricsClientProtocol = Depends(get_lyrics_client),
) -> LyricsSearchResponse:
    """Find songs by a lyrics fragment. Searches the full lyrics text of our
    own trained catalog first (real full-text match), then falls back to
    lrclib's keyword search — whose `q` matches title/artist metadata, not
    lyric content — for songs we haven't trained yet."""
    needle = " ".join(q.lower().split())
    results: list[LyricsSearchResult] = []
    seen: set[tuple[str, str]] = set()

    for song in db.scalars(select(Song)).all():
        if song.lyrics and needle in " ".join(song.lyrics.lower().split()):
            results.append(
                LyricsSearchResult(
                    title=song.title,
                    artist=song.artist,
                    album=None,
                    snippet=_snippet_around(song.lyrics, needle),
                    song_id=str(song.id),
                    album_art_url=song.album_art_url,
                )
            )
            seen.add((song.title.lower(), song.artist.lower()))

    if len(results) < limit:
        try:
            hits = lyrics_client.search(q, limit=limit)
        except httpx.HTTPError:
            hits = []
        for h in hits:
            if (h.title.lower(), h.artist.lower()) in seen:
                continue
            results.append(
                LyricsSearchResult(
                    title=h.title,
                    artist=h.artist,
                    album=h.album,
                    snippet=_snippet(h.plain_lyrics),
                )
            )

    return LyricsSearchResponse(results=results[:limit])


@router.get("/lyrics", response_model=LyricsResponse)
def get_lyrics(
    title: str = Query(..., min_length=1),
    artist: str = Query(..., min_length=1),
    lyrics_client: LyricsClientProtocol = Depends(get_lyrics_client),
) -> LyricsResponse:
    try:
        hit = lyrics_client.get_lyrics(title, artist)
    except httpx.HTTPError:
        hit = None
    if hit is None:
        return LyricsResponse(title=None, artist=None, lyrics=None)
    return LyricsResponse(title=hit.title, artist=hit.artist, lyrics=hit.plain_lyrics)


@router.get("/preview", response_model=PreviewResponse)
def get_preview(
    title: str = Query(..., min_length=1),
    artist: str = Query(..., min_length=1),
    preview_client: PreviewClientProtocol = Depends(get_preview_client),
) -> PreviewResponse:
    try:
        hit = preview_client.find_preview(title, artist)
    except httpx.HTTPError:
        hit = None
    if hit is None:
        return PreviewResponse(preview_url=None, artwork_url=None)
    return PreviewResponse(preview_url=hit.preview_url, artwork_url=hit.artwork_url)


def _decade_of(release_year: int | None) -> str | None:
    if release_year is None:
        return None
    return f"{(release_year // 10) * 10}s"


@router.get("/filter", response_model=FilterResponse)
def filter_songs(
    artist: str | None = Query(None),
    genre: str | None = Query(None),
    decade: str | None = Query(None),
    tempo: str | None = Query(None),
    db: Session = Depends(get_db),
) -> FilterResponse:
    """Filter the local (trained) catalog by feature. All filters are ANDed.
    The catalog is small enough to filter in Python; revisit if it grows."""
    songs = db.scalars(select(Song)).all()

    def keep(song: Song) -> bool:
        if artist and artist.lower() not in song.artist.lower():
            return False
        if genre and not any(genre.lower() in g.lower() for g in (song.genres or [])):
            return False
        if decade and _decade_of(song.release_year) != decade:
            return False
        if tempo and song.tempo_bucket != tempo:
            return False
        return True

    kept = sorted((s for s in songs if keep(s)), key=lambda s: (s.artist, s.title))
    return FilterResponse(results=[SongOut.from_song(s) for s in kept])


@router.get("/facets", response_model=FacetsResponse)
def get_facets(db: Session = Depends(get_db)) -> FacetsResponse:
    """Distinct filter values present in the trained catalog, for the
    find-by-features UI to offer as options."""
    songs = db.scalars(select(Song)).all()
    genres = sorted({g for s in songs for g in (s.genres or [])})
    decades = sorted({d for s in songs if (d := _decade_of(s.release_year))})
    tempos = sorted({s.tempo_bucket for s in songs if s.tempo_bucket})
    artists = sorted({s.artist for s in songs})
    return FacetsResponse(genres=genres, decades=decades, tempos=tempos, artists=artists)
