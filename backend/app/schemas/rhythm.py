from typing import Literal

from pydantic import field_validator

from app.config import settings
from app.schemas.song import CamelModel, SongOut


def _validate_taps(taps: list[int], min_len: int) -> list[int]:
    if len(taps) < min_len:
        raise ValueError(f"at least {min_len} taps are required")
    if len(taps) > settings.max_tap_count:
        raise ValueError(f"at most {settings.max_tap_count} taps are allowed")
    if taps[0] != 0:
        raise ValueError("tapsMs must be relative to the first tap (tapsMs[0] == 0)")
    for i in range(1, len(taps)):
        if taps[i] <= taps[i - 1]:
            raise ValueError("tapsMs must be strictly increasing")
    if taps[-1] > settings.max_tap_span_ms:
        raise ValueError(f"tap sequence spans more than {settings.max_tap_span_ms}ms")
    return taps


class CandidateSong(SongOut):
    match: int


class QueryRequest(CamelModel):
    taps_ms: list[int]

    @field_validator("taps_ms")
    @classmethod
    def _check_taps(cls, v: list[int]) -> list[int]:
        return _validate_taps(v, settings.min_taps_for_query)


class QueryResponse(CamelModel):
    status: Literal["confident", "ambiguous", "no_match"]
    match: CandidateSong | None
    candidates: list[CandidateSong]
    query_log_id: str | None = None


class TrainingRequest(CamelModel):
    query: str
    taps_ms: list[int]

    @field_validator("query")
    @classmethod
    def _check_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("taps_ms")
    @classmethod
    def _check_taps(cls, v: list[int]) -> list[int]:
        return _validate_taps(v, settings.min_taps_for_training)


class TrainingResponse(CamelModel):
    status: Literal["resolved", "not_found"]
    song: SongOut | None
    message: str | None


class ConfirmRequest(CamelModel):
    song_id: str
    taps_ms: list[int]

    @field_validator("taps_ms")
    @classmethod
    def _check_taps(cls, v: list[int]) -> list[int]:
        return _validate_taps(v, settings.min_taps_for_training)


class ConfirmResponse(CamelModel):
    status: Literal["ok"]
    sample_count: int


class FeedbackRequest(CamelModel):
    query_log_id: str
    outcome: Literal["saved", "not_it"]
    song_id: str | None = None  # required when outcome == 'saved'


class FeedbackResponse(CamelModel):
    status: Literal["ok"]
