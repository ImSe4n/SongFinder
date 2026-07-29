# SongFinder backend

Song identification by rhythm and by sound:

- **Rhythm** — a "Query by Tapping" engine: tempo-invariant log-ratio
  interval sequences matched via banded DTW against a crowdsourced corpus of
  user-submitted tap sequences.
- **Audio** — a Shazam-style fingerprinting engine (Wang 2003): spectrogram
  constellation peaks -> anchor/target pair hashes -> offset-histogram
  matching. **Coverage is what users have taught it**: there is no licensed
  commercial fingerprint database, so a song is only recognizable after
  someone has recorded it once via the teach flow. Same self-improving
  philosophy as the rhythm engine.

Plus: lyrics-fragment search and full lyrics via the free lrclib.net API (no
key needed), 30-second playback previews via the free iTunes Search API (no
key needed), catalog filter search, and a thin Spotify integration for song
metadata.

## Setup

```
cd backend
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

cp .env.example .env
# fill in SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET
```

Spotify uses the **Client Credentials flow** only (server-to-server) — you
need a Client ID and Client Secret from the Spotify Developer Dashboard, but
**no redirect URI**. Redirect URIs are only required for user-login
(Authorization Code) flows, which this backend never uses; it only calls the
public Search and Artists endpoints to look up song metadata.

## Run

```
./.venv/Scripts/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is required, not optional, if you want to hit this from a
physical device running the Expo app over Wi-Fi — `localhost` only resolves
to the dev machine itself. Swagger docs at `http://localhost:8000/docs`.

## Test

```
./.venv/Scripts/pytest -q
```

Runs entirely offline against an in-memory SQLite database with a fake
Spotify client — no real credentials or network access needed.

## Endpoints

- `POST /rhythm/query` — `{ tapsMs: number[] }` → best match (if confident)
  plus a ranked candidate list.
- `POST /rhythm/samples` — `{ query: string, tapsMs: number[] }` — "teach" a
  known song's rhythm. Resolves the song via the local catalog first, then
  Spotify if not already known.
- `POST /rhythm/confirm` — `{ songId: string, tapsMs: number[] }` — adds a
  confirmed-correct sample after a real identification, reinforcing the
  corpus.
- `POST /audio/query` — multipart audio upload → fingerprint match against
  the taught-audio corpus (confident match or candidates).
- `POST /audio/samples` — multipart `file` + `query` form field — teach a
  song by playing it out loud while recording.
- `GET /songs/lyrics-search?q=` — find songs by a lyrics fragment (lrclib).
- `GET /songs/lyrics?title=&artist=` — full lyrics for a song (lrclib).
- `GET /songs/preview?title=&artist=` — 30s playback preview URL + high-res
  artwork (iTunes Search API).
- `GET /songs/filter?artist=&genre=&decade=&tempo=` — filter the trained
  catalog by features (all params optional, ANDed).
- `GET /songs/facets` — distinct genre/decade/tempo/artist values in the
  catalog, for filter UIs.
- `GET /songs/search?q=` — stateless Spotify search passthrough. Not yet
  wired into any frontend UI.

## How matching works

See `app/rhythm/features.py` and `app/rhythm/matching.py`. Short version:
tap timestamps → inter-onset intervals → a tempo-invariant log-ratio
sequence → compared against every stored sample via banded/capped/
length-normalized DTW → best-matching sample per song → confidence-ranked
list → a confident top result requires clearing both a confidence threshold
and a margin over 2nd place, otherwise it's returned as an ambiguous
candidate list.

All thresholds live in `.env` (see `.env.example`) and are placeholder
starting values — expect to tune `MATCH_CONFIDENCE_THRESHOLD`,
`MATCH_MARGIN_THRESHOLD`, `DTW_ALPHA`, `DTW_BAND_FRACTION`, and
`DTW_COST_CAP` once there's real usage data.

A song only ever enters the catalog alongside its first rhythm sample —
there's no such thing as a 0-sample song, so matching never has to special
case it. A song with exactly one sample can still be matched, just less
reliably; `sampleCount` is included in every song response so the frontend
could eventually surface "lightly trained" songs differently.

## Known limitations / not done here

- **No auth or rate-limiting.** Anyone with the URL can submit training
  samples or burn Spotify API quota. Fine for local development; needs
  addressing before any public deployment.
- **Brute-force matching.** `/rhythm/query` runs DTW against every stored
  sample for every song. Fine for a small early corpus. If the corpus grows
  large, look at: matching against one representative sample per song
  instead of all of them, a cheap tempo-bucket pre-filter before running
  DTW, or eventually a fixed-length embedding + ANN index (e.g. FAISS).
- **No database migrations.** `Base.metadata.create_all()` runs on startup,
  which is fine for a greenfield SQLite app. Add Alembic when moving to
  Postgres.
- **Genre is `genres[0]`** from Spotify's unordered artist genre list — not
  precise, acceptable for MVP.
- **Mood is not populated.** Spotify deprecated the Audio Features endpoint
  for new API apps, so there's no reliable mood signal available; the field
  was dropped from the app rather than faked.

## Moving to Postgres later

Change `DATABASE_URL` in `.env` (e.g.
`postgresql+psycopg://user:pass@host/db`), install a driver (`psycopg` or
`psycopg2-binary`), and add Alembic for migrations. `app/database.py`
isolates the engine/session setup so this is a config change, not a rewrite.
