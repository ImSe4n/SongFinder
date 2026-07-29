from app.services.catalog import get_or_create_song, search_local_catalog
from app.services.spotify import ResolvedTrack


def _make_song(db, spotify_id: str, title: str, artist: str):
    return get_or_create_song(
        db,
        ResolvedTrack(
            spotify_track_id=spotify_id,
            title=title,
            artist=artist,
            genres=[],
            release_year=None,
            album_art_url=None,
        ),
    )


def test_search_local_catalog_does_not_confuse_unrelated_songs(db_session):
    """Regression test for the exact real-world collision found importing
    QBT-Extended: querying "Ben E. King Stand By Me" (artist+title, no
    separator other than the word "by" inside the title itself) must NOT
    match "American Pie by Don McLean" just because normalized "american"
    happens to contain the substring "me"."""
    _make_song(db_session, "sp1", "American Pie", "Don McLean")
    db_session.commit()

    result = search_local_catalog(db_session, "Ben E. King Stand By Me")
    assert result is None


def test_search_local_catalog_still_finds_structured_match_with_title_containing_by(db_session):
    """A title containing the word 'by' must still resolve correctly when
    the query uses an explicit separator (the case the feature actually
    supports) — only the *unseparated* "artist title" ordering was ever
    exploitable via the short-fragment bug, and that ordering was never a
    legitimate match path in the first place (haystack is always
    "title artist" order; an unseparated reordered query has no separator
    telling the matcher which words are the title vs. the artist)."""
    song = _make_song(db_session, "sp1", "Stand By Me", "Ben E. King")
    db_session.commit()

    result = search_local_catalog(db_session, "Stand By Me - Ben E. King")
    assert result is not None
    assert result.id == song.id


def test_search_local_catalog_matches_title_dash_artist_style_query(db_session):
    song = _make_song(db_session, "sp1", "Bohemian Rhapsody", "Queen")
    db_session.commit()

    result = search_local_catalog(db_session, "Bohemian Rhapsody - Queen")
    assert result is not None
    assert result.id == song.id


def test_search_local_catalog_returns_none_when_nothing_matches(db_session):
    _make_song(db_session, "sp1", "Yesterday", "The Beatles")
    db_session.commit()

    assert search_local_catalog(db_session, "Some Totally Different Song") is None
