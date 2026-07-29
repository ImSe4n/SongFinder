from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from app.models import Song


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


def _decade(release_year: int | None) -> str:
    if release_year is None:
        return "Unknown"
    return f"{(release_year // 10) * 10}s"


class SongOut(CamelModel):
    id: str
    title: str
    artist: str
    genre: str
    decade: str
    tempo: str | None
    spotify_track_id: str
    album_art_url: str | None
    sample_count: int
    audio_sample_count: int = 0

    @classmethod
    def from_song(cls, song: Song) -> "SongOut":
        return cls(
            id=str(song.id),
            title=song.title,
            artist=song.artist,
            genre=song.genres[0] if song.genres else "Unknown",
            decade=_decade(song.release_year),
            tempo=song.tempo_bucket,
            spotify_track_id=song.spotify_track_id,
            album_art_url=song.album_art_url,
            sample_count=song.sample_count,
            audio_sample_count=song.audio_sample_count,
        )


class SongSearchResult(CamelModel):
    spotify_track_id: str
    title: str
    artist: str
    album_art_url: str | None


class SongSearchResponse(CamelModel):
    results: list[SongSearchResult]


class LyricsSearchResult(CamelModel):
    title: str
    artist: str
    album: str | None
    snippet: str | None  # the matching lines of the lyrics, for the results list
    song_id: str | None = None  # set when the hit is a song in our own catalog
    album_art_url: str | None = None


class LyricsSearchResponse(CamelModel):
    results: list[LyricsSearchResult]


class LyricsResponse(CamelModel):
    title: str | None
    artist: str | None
    lyrics: str | None


class PreviewResponse(CamelModel):
    preview_url: str | None
    artwork_url: str | None


class FacetsResponse(CamelModel):
    genres: list[str]
    decades: list[str]
    tempos: list[str]
    artists: list[str]


class FilterResponse(CamelModel):
    results: list[SongOut]
