import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine, ensure_schema, get_session_factory
from app.routers import audio, rhythm, songs

Base.metadata.create_all(bind=engine)
ensure_schema()


def _warm_audio_fingerprint_cache(session_factory) -> None:
    from app.services.catalog import warm_fingerprint_cache

    with session_factory() as db:
        warm_fingerprint_cache(db)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Pre-load the rhythm model and pre-embed the catalog in a background
    # thread, so the first real /rhythm/query doesn't pay the ~15s cold-start.
    # Backgrounded so it never delays the server becoming ready.
    from app.rhythm.hybrid_matching import warm_cache

    threading.Thread(
        target=warm_cache, args=(get_session_factory(),), daemon=True
    ).start()
    # Same idea for /audio/query: the audio_fingerprints table is 30M+ rows,
    # and the first lookup after process start pays a one-time OS-page-cache
    # cold-read cost (measured ~7s cold vs ~1s warm) — pay it here instead of
    # on a user's first Listen-mode query.
    threading.Thread(
        target=_warm_audio_fingerprint_cache,
        args=(get_session_factory(),),
        daemon=True,
    ).start()
    yield


app = FastAPI(title="SongFinder API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rhythm.router)
app.include_router(songs.router)
app.include_router(audio.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
