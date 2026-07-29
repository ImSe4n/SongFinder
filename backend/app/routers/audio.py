import httpx
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session, sessionmaker

from app.audio.decode import TARGET_SAMPLE_RATE, AudioDecodeError, decode_to_mono
from app.audio.fingerprint import fingerprint
from app.audio.matching import decide_audio_match, rank_audio_matches
from app.audio.onsets import extract_tap_times_vocal_aware
from app.config import settings
from app.database import get_db, get_session_factory
from app.models import Song
from app.schemas.rhythm import CandidateSong, ConfirmResponse, QueryResponse, TrainingResponse
from app.schemas.song import SongOut
from app.services.catalog import (
    attach_lyrics,
    get_or_create_song,
    lookup_fingerprints,
    record_audio_fingerprints,
    record_sample,
    search_local_catalog,
)
from app.services.lyrics import LyricsClientProtocol, get_lyrics_client
from app.services.query_log import log_audio_query
from app.services.spotify import SpotifyClientProtocol, get_spotify_client

router = APIRouter(prefix="/audio", tags=["audio"])


async def _read_and_fingerprint(file: UploadFile):
    """Returns (raw_bytes, decoded_samples, fingerprint_hashes) or raises
    HTTP errors."""
    data = await file.read()
    if len(data) > settings.audio_max_upload_bytes:
        raise HTTPException(status_code=413, detail="Audio upload too large")
    if not data:
        raise HTTPException(status_code=422, detail="Empty audio upload")

    try:
        samples = decode_to_mono(data)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if samples.size < settings.audio_min_clip_seconds * TARGET_SAMPLE_RATE:
        raise HTTPException(
            status_code=422,
            detail=f"Clip too short — record at least {settings.audio_min_clip_seconds:g}s",
        )

    hashes = fingerprint(samples)
    if not hashes:
        raise HTTPException(
            status_code=422,
            detail="Couldn't extract any audio features — was the clip silent?",
        )
    return data, samples, hashes


def _record_auto_rhythm_task(
    song_id: int, samples: np.ndarray, data: bytes, session_factory: sessionmaker
) -> None:
    """Experimental: derive a human-like tap sequence from the taught audio
    and store it as a quarantined source='auto' rhythm sample.

    Runs as a FastAPI background task, AFTER the HTTP response has already
    gone out — vocal-aware extraction can take real time (Demucs on CPU),
    and the mobile app should never wait on it. Takes `session_factory`
    (from Depends(get_session_factory), NOT a bare SessionLocal import) so
    tests can override it to the same isolated database `get_db` uses —
    otherwise this would silently write to the real configured database
    during a test run.
    """
    if not settings.auto_rhythm_enabled:
        return
    try:
        with session_factory() as db:
            song = db.get(Song, song_id)
            if song is None:
                return
            taps = extract_tap_times_vocal_aware(samples, data)
            if len(taps) >= settings.min_taps_for_training:
                record_sample(db, song, taps, source="auto")
                db.commit()
    except Exception:
        pass


@router.post("/query", response_model=QueryResponse)
async def query_audio(file: UploadFile, db: Session = Depends(get_db)) -> QueryResponse:
    _, _, hashes = await _read_and_fingerprint(file)

    db_hits = lookup_fingerprints(db, [h for h, _ in hashes])
    ranked = rank_audio_matches(hashes, db_hits)
    status, top = decide_audio_match(ranked)

    top_n = ranked[: settings.max_candidates]
    songs = {
        s.id: s
        for s in db.query(Song).filter(Song.id.in_([m.song_id for m in top_n])).all()
    }
    candidates = [
        CandidateSong(
            **SongOut.from_song(songs[m.song_id]).model_dump(),
            match=round(m.confidence),
        )
        for m in top_n
        if m.song_id in songs
    ]

    match = None
    if status == "confident" and top is not None:
        match = next((c for c in candidates if c.id == str(top.song_id)), None)
        # Self-improving corpus: a confident audio match already cleared the
        # aligned-hash margin check against the runner-up, so it's safe to
        # fold this clip's fingerprints into the song's stored set without
        # asking — same trust bar as the rhythm side (see routers/rhythm.py).
        # Capped (unlike rhythm) since each clip adds ~150-200 rows to a
        # multi-million-row table — see auto_confident_audio_sample_cap.
        matched_song = songs.get(top.song_id)
        if matched_song is not None and (
            matched_song.audio_sample_count < settings.auto_confident_audio_sample_cap
        ):
            record_audio_fingerprints(db, matched_song, hashes)

    log_entry = log_audio_query(
        db,
        hash_count=len(hashes),
        status=status,
        top_song_id=top.song_id if top is not None else None,
        top_confidence=top.confidence if top is not None else None,
        candidate_count=len(candidates),
    )
    db.commit()

    return QueryResponse(
        status=status, match=match, candidates=candidates, query_log_id=str(log_entry.id)
    )


@router.post("/samples", response_model=TrainingResponse)
async def submit_audio_sample(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    query: str = Form(..., min_length=1),
    db: Session = Depends(get_db),
    session_factory: sessionmaker = Depends(get_session_factory),
    spotify_client: SpotifyClientProtocol = Depends(get_spotify_client),
    lyrics_client: LyricsClientProtocol = Depends(get_lyrics_client),
) -> TrainingResponse:
    data, samples, hashes = await _read_and_fingerprint(file)

    song = search_local_catalog(db, query)
    if song is None:
        try:
            resolved = spotify_client.resolve_track(query)
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
                message=f"Couldn't find a song matching \"{query}\" on Spotify.",
            )
        song = get_or_create_song(db, resolved)

    record_audio_fingerprints(db, song, hashes)
    attach_lyrics(db, song, lyrics_client)
    db.commit()
    db.refresh(song)

    # Slow (vocal separation can take real time on CPU) — runs after the
    # response is sent, never makes the app wait.
    background_tasks.add_task(_record_auto_rhythm_task, song.id, samples, data, session_factory)

    return TrainingResponse(status="resolved", song=SongOut.from_song(song), message=None)


@router.post("/confirm", response_model=ConfirmResponse)
async def confirm_audio(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    song_id: str = Form(...),
    db: Session = Depends(get_db),
    session_factory: sessionmaker = Depends(get_session_factory),
) -> ConfirmResponse:
    """Reinforce the audio corpus after a confirmed-correct identification —
    the audio-mode counterpart of POST /rhythm/confirm."""
    data, samples, hashes = await _read_and_fingerprint(file)

    try:
        song_pk = int(song_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Song not found")
    song = db.get(Song, song_pk)
    if song is None:
        raise HTTPException(status_code=404, detail="Song not found")

    record_audio_fingerprints(db, song, hashes)
    db.commit()
    db.refresh(song)

    background_tasks.add_task(_record_auto_rhythm_task, song.id, samples, data, session_factory)

    return ConfirmResponse(status="ok", sample_count=song.audio_sample_count)
