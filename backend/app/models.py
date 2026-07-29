from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Song(Base):
    __tablename__ = "songs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    spotify_track_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    title: Mapped[str] = mapped_column(String)
    artist: Mapped[str] = mapped_column(String)
    genres: Mapped[list[str]] = mapped_column(JSON, default=list)
    release_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    album_art_url: Mapped[str | None] = mapped_column(String, nullable=True)

    tempo_bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    tempo_bucket: Mapped[str | None] = mapped_column(String, nullable=True)

    # Full plain lyrics (from lrclib), fetched once at training time so
    # lyrics-fragment search can run locally over text we actually have.
    lyrics: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Filename this song was bulk-trained from (scripts/bulk_train.py). Used
    # as the authoritative "already trained this file?" check — title/artist
    # text can change on re-resolve (e.g. --no-spotify placeholder -> real
    # Spotify metadata), which would otherwise make the same file look new.
    source_file: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    audio_sample_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    samples: Mapped[list["RhythmSample"]] = relationship(
        back_populates="song", cascade="all, delete-orphan"
    )
    fingerprints: Mapped[list["AudioFingerprint"]] = relationship(
        back_populates="song", cascade="all, delete-orphan"
    )


class RhythmSample(Base):
    __tablename__ = "rhythm_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), index=True)

    raw_taps_ms: Mapped[list[int]] = mapped_column(JSON)
    ioi_ms: Mapped[list[int]] = mapped_column(JSON)
    log_ratio_vector: Mapped[list[float]] = mapped_column(JSON)

    source: Mapped[str] = mapped_column(String)
    # 'train' | 'confirm' (app-native, from real SongFinder users) |
    # 'auto_confident' (unprompted — a /rhythm/query that the system itself
    # was confident about, auto-stored as a new sample; see routers/rhythm.py.
    # Distinct from 'confirm' since no human verified it, kept separate so it
    # can be filtered out later if it turns out to hurt quality) |
    # 'auto' (machine-guessed from taught audio, quarantined — see
    # rhythm/matching.py) |
    # 'external_qbte' (real human taps imported from the QBT-Extended
    # research dataset — see scripts/import_qbt_extended.py; CC BY-NC-SA,
    # non-commercial only, kept as a distinct source so it can be filtered
    # or removed if SongFinder's licensing situation ever requires it)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    song: Mapped["Song"] = relationship(back_populates="samples")


class AudioFingerprint(Base):
    """One spectral-peak-pair hash from a taught audio clip. A single clip
    produces thousands of these; matching looks hashes up by value and
    histograms time offsets (see app/audio/matching.py)."""

    __tablename__ = "audio_fingerprints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), index=True)
    hash: Mapped[int] = mapped_column(Integer, index=True)
    offset: Mapped[int] = mapped_column(Integer)  # anchor time, in STFT frames

    song: Mapped["Song"] = relationship(back_populates="fingerprints")


class QueryLog(Base):
    """One identification attempt (Tap or Listen) and what the user did with
    the result. This is the actual prerequisite dataset for anything ML down
    the line: real (query, outcome) pairs, not machine-generated guesses.

    Privacy: rhythm queries store the tap intervals themselves (that *is*
    the data — just numbers, no audio). Audio queries never store the
    recording, only a summary (hash/duration counts).
    """

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_type: Mapped[str] = mapped_column(String)  # 'rhythm' | 'audio'

    # What the engine decided
    status: Mapped[str] = mapped_column(String)  # 'confident' | 'ambiguous' | 'no_match'
    top_song_id: Mapped[int | None] = mapped_column(ForeignKey("songs.id"), nullable=True)
    top_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0)

    # Raw query data — tap intervals for rhythm queries (this literally *is*
    # future training data); just summary stats for audio (no audio stored).
    taps_ms: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    audio_hash_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # What the user actually did with the result — filled in later via
    # POST /feedback. This is the label.
    outcome: Mapped[str | None] = mapped_column(String, nullable=True)
    # 'saved' | 'not_it' | null (no feedback yet)
    outcome_song_id: Mapped[int | None] = mapped_column(ForeignKey("songs.id"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    responded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
