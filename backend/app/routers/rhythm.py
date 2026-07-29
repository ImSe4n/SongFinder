import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import QueryLog, Song
from app.rhythm.features import iois_to_log_ratio, taps_to_iois
from app.rhythm.hybrid_matching import HybridUnavailable, rank_hybrid
from app.rhythm.matching import decide_match, rank_candidates
from app.schemas.rhythm import (
    CandidateSong,
    ConfirmRequest,
    ConfirmResponse,
    FeedbackRequest,
    FeedbackResponse,
    QueryRequest,
    QueryResponse,
    TrainingRequest,
    TrainingResponse,
)
from app.schemas.song import SongOut
from app.services.catalog import (
    attach_lyrics,
    get_or_create_song,
    list_matchable_songs,
    record_sample,
    search_local_catalog,
)
from app.services.lyrics import LyricsClientProtocol, get_lyrics_client
from app.services.query_log import log_rhythm_query, record_feedback
from app.services.spotify import SpotifyClientProtocol, get_spotify_client

router = APIRouter(prefix="/rhythm", tags=["rhythm"])


def _match_dtw_only(taps_ms: list[int], songs, songs_by_id):
    """Pure-DTW ranking (the original path). Returns
    (status, candidates, top_song_id, top_confidence)."""
    query_r = iois_to_log_ratio(taps_to_iois(taps_ms))
    songs_with_vectors = [
        (
            song.id,
            [s.log_ratio_vector for s in song.samples if s.source != "auto"],
            [s.log_ratio_vector for s in song.samples if s.source == "auto"],
        )
        for song in songs
    ]
    ranked = rank_candidates(query_r, songs_with_vectors)
    status, top = decide_match(ranked)
    top_n = ranked[: settings.max_candidates]
    candidates = [
        CandidateSong(
            **SongOut.from_song(songs_by_id[r.song_id]).model_dump(),
            match=round(r.confidence),
        )
        for r in top_n
    ]
    return status, candidates, (top.song_id if top else None), (top.confidence if top else None)


def _match_hybrid(taps_ms: list[int], songs, songs_by_id):
    """Learned-encoder + DTW routed fusion. Raises HybridUnavailable if the
    model can't be used, so the caller can fall back to DTW."""
    status, fused_ids, confidence, _model_top, _dtw_top = rank_hybrid(taps_ms, songs)
    top_ids = fused_ids[: settings.max_candidates]
    candidates = [
        CandidateSong(
            **SongOut.from_song(songs_by_id[sid]).model_dump(),
            match=confidence.get(sid, 0),
        )
        for sid in top_ids
        if sid in songs_by_id
    ]
    top_song_id = fused_ids[0] if fused_ids else None
    top_conf = float(confidence.get(top_song_id, 0)) if top_song_id is not None else None
    return status, candidates, top_song_id, top_conf


@router.post("/query", response_model=QueryResponse)
def query_rhythm(body: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    songs = list_matchable_songs(db)
    songs_by_id = {song.id: song for song in songs}

    used_hybrid = False
    if settings.hybrid_matching_enabled:
        try:
            status, candidates, top_song_id, top_conf = _match_hybrid(
                body.taps_ms, songs, songs_by_id
            )
            used_hybrid = True
        except HybridUnavailable:
            used_hybrid = False  # transparently fall back to DTW

    if not used_hybrid:
        status, candidates, top_song_id, top_conf = _match_dtw_only(
            body.taps_ms, songs, songs_by_id
        )

    match = None
    if status == "confident" and top_song_id is not None:
        match = next((c for c in candidates if c.id == str(top_song_id)), None)
        # Self-improving corpus: a confident match already required two
        # independent matchers (DTW + the learned encoder, when available) to
        # agree — a stronger trust signal than the explicit /confirm click,
        # so it's safe to store this query as a new sample without asking.
        # Kept as a distinct source (not 'confirm') so it can be filtered out
        # later if it turns out to hurt quality.
        record_sample(db, songs_by_id[top_song_id], body.taps_ms, source="auto_confident")

    log_entry = log_rhythm_query(
        db,
        taps_ms=body.taps_ms,
        status=status,
        top_song_id=top_song_id,
        top_confidence=top_conf,
        candidate_count=len(candidates),
    )
    db.commit()

    return QueryResponse(
        status=status, match=match, candidates=candidates, query_log_id=str(log_entry.id)
    )


@router.post("/samples", response_model=TrainingResponse)
def submit_training_sample(
    body: TrainingRequest,
    db: Session = Depends(get_db),
    spotify_client: SpotifyClientProtocol = Depends(get_spotify_client),
    lyrics_client: LyricsClientProtocol = Depends(get_lyrics_client),
) -> TrainingResponse:
    song = search_local_catalog(db, body.query)

    if song is None:
        try:
            resolved = spotify_client.resolve_track(body.query)
        except httpx.HTTPError:
            return TrainingResponse(
                status="not_found",
                song=None,
                message="Couldn't reach Spotify to look that song up — try again in a moment.",
            )
        if resolved is None:
            return TrainingResponse(
                status="not_found",
                song=None,
                message=f"Couldn't find a song matching \"{body.query}\" on Spotify.",
            )
        song = get_or_create_song(db, resolved)

    record_sample(db, song, body.taps_ms, source="train")
    attach_lyrics(db, song, lyrics_client)
    db.commit()
    db.refresh(song)

    return TrainingResponse(status="resolved", song=SongOut.from_song(song), message=None)


@router.post("/confirm", response_model=ConfirmResponse)
def confirm_sample(body: ConfirmRequest, db: Session = Depends(get_db)) -> ConfirmResponse:
    try:
        song_pk = int(body.song_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Song not found")

    song = db.get(Song, song_pk)
    if song is None:
        raise HTTPException(status_code=404, detail="Song not found")

    record_sample(db, song, body.taps_ms, source="confirm")
    db.commit()
    db.refresh(song)

    return ConfirmResponse(status="ok", sample_count=song.sample_count)


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackRequest, db: Session = Depends(get_db)) -> FeedbackResponse:
    """Records what the user did with a query result — the label half of
    the (query, outcome) pair that makes this data useful later. Handles
    feedback for BOTH /rhythm/query and /audio/query (QueryLog isn't
    type-specific), so the frontend calls this one endpoint either way."""
    try:
        log_id = int(body.query_log_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Query log not found")

    entry = db.get(QueryLog, log_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Query log not found")

    song_pk: int | None = None
    if body.outcome == "saved":
        if body.song_id is None:
            raise HTTPException(status_code=422, detail="songId is required when outcome is 'saved'")
        try:
            song_pk = int(body.song_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Song not found")

    record_feedback(db, entry, outcome=body.outcome, song_id=song_pk)
    db.commit()

    return FeedbackResponse(status="ok")
