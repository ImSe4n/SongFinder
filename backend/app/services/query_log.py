"""Logs every identification attempt and, later, what the user did with the
result. This is the actual prerequisite dataset for any future ML work:
real (query, outcome) pairs from real usage, as opposed to machine-generated
guesses. See scripts/training_status.py for where this stands."""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import QueryLog


def log_rhythm_query(
    db: Session,
    taps_ms: list[int],
    status: str,
    top_song_id: int | None,
    top_confidence: float | None,
    candidate_count: int,
) -> QueryLog:
    entry = QueryLog(
        query_type="rhythm",
        status=status,
        top_song_id=top_song_id,
        top_confidence=top_confidence,
        candidate_count=candidate_count,
        taps_ms=taps_ms,
    )
    db.add(entry)
    db.flush()
    return entry


def log_audio_query(
    db: Session,
    hash_count: int,
    status: str,
    top_song_id: int | None,
    top_confidence: float | None,
    candidate_count: int,
) -> QueryLog:
    entry = QueryLog(
        query_type="audio",
        status=status,
        top_song_id=top_song_id,
        top_confidence=top_confidence,
        candidate_count=candidate_count,
        audio_hash_count=hash_count,
    )
    db.add(entry)
    db.flush()
    return entry


def record_feedback(
    db: Session, entry: QueryLog, outcome: str, song_id: int | None
) -> None:
    entry.outcome = outcome
    entry.outcome_song_id = song_id
    entry.responded_at = datetime.now(timezone.utc)
    db.flush()
