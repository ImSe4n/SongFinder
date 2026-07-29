from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    spotify_client_id: str = ""
    spotify_client_secret: str = ""

    database_url: str = "sqlite:///./songfinder.db"

    match_confidence_threshold: float = 70
    match_margin_threshold: float = 15
    dtw_alpha: float = 1.5
    dtw_band_fraction: float = 0.2
    dtw_cost_cap: float = 4.0
    max_candidates: int = 8

    min_taps_for_query: int = 5
    min_taps_for_training: int = 4
    max_tap_count: int = 100
    max_tap_span_ms: int = 30_000
    min_ioi_ms: int = 40

    tempo_slow_max_bpm: float = 90
    tempo_fast_min_bpm: float = 130

    # Audio fingerprint matching
    #
    # Recalibrated against real query_logs data (2026-07-22): the only two
    # queries with confirmed-correct feedback ('saved') aligned at just 9 and
    # 16 hashes out of 400/834 total query hashes (~2% alignment) — real
    # phone-mic recordings of playback (room noise, compression, mic
    # response) diverge from the clean audio the catalog was fingerprinted
    # from far more than a lab recording would. Both were rejected as
    # "ambiguous" under the old min_hashes=20 floor despite being correct.
    # min_hashes is now just a sanity floor against tiny clips; the margin
    # ratio (below) is what actually discriminates real matches from noise,
    # since a spurious spike would need many hash collisions landing on the
    # exact same time offset by chance, which margin_ratio catches regardless
    # of the absolute count.
    audio_match_min_hashes: int = 8  # aligned hashes needed for a confident match
    audio_noise_floor_hashes: int = 4  # below this, treat as no match at all
    audio_match_margin_ratio: float = 1.6  # top must beat runner-up by this factor
    audio_max_upload_bytes: int = 15_000_000
    audio_min_clip_seconds: float = 2.0

    # Experimental: auto-derive rhythm (tap) samples from taught audio.
    # Auto samples are quarantined — the distance penalty keeps them from ever
    # outranking human-taught rhythms (they can only surface candidates).
    auto_rhythm_enabled: bool = True
    auto_rhythm_distance_penalty: float = 0.35

    # Self-improving corpus: a confident /query result (rhythm or audio) is
    # automatically folded back in as a new training sample — no explicit
    # /confirm needed, since "confident" already required independent
    # matchers to agree. Rhythm samples are cheap (one small row), so no cap;
    # audio fingerprints are NOT — each clip adds ~150-200 rows to a
    # multi-million-row table, and near-duplicate fingerprints for the same
    # song fight the hash-popularity cap in lookup_fingerprints() rather than
    # adding real signal. Cap how many auto-confident audio samples a song
    # can accumulate; rhythm samples always count toward a strong signal so
    # they're uncapped.
    auto_confident_audio_sample_cap: int = 5

    # Vocal isolation (Demucs) before onset detection — see
    # app/audio/vocal_separation.py. Real accuracy win (measured via
    # scripts/eval_auto_rhythm.py) but genuinely slow on CPU (~seconds to
    # tens of seconds per song), which is why it always runs on a short
    # hook-window slice, not a whole track, and why the live API routes run
    # it via a background task rather than inline in the request.
    vocal_separation_enabled: bool = True

    # Hybrid rhythm matching: fuse the learned encoder (app/ml/rhythm_encoder_v2.py,
    # a SupCon/TCN metric-learning model trained on human tap data) with the
    # DTW matchers, routed by query length. Benchmarked (scripts/eval_matchers_v2.py,
    # eval_full_catalog.py) as beating DTW-alone, especially on short queries.
    # When disabled or the checkpoint is missing/unloadable, the endpoint
    # transparently falls back to the pure-DTW path — the model is strictly
    # additive, never a hard dependency.
    #
    # NOTE: the shipped checkpoint is trained substantially on QBT-Extended
    # (CC BY-NC-SA, non-commercial). Fine for personal/non-commercial use;
    # resolve licensing before any commercial deployment.
    hybrid_matching_enabled: bool = True
    rhythm_model_path: str = "models/rhythm_encoder_v2.pt"
    # Queries with <= this many taps route through subsequence-DTW (better for
    # short fragments); longer queries use global DTW. Fusion adds the model on
    # top in both cases. 15 taps ~ the boundary between "tapped a fragment" and
    # "tapped the whole hook" seen in real usage.
    hybrid_short_query_taps: int = 15
    hybrid_rrf_k: int = 60
    # Two-stage retrieval: the (fast) model embedding filters the catalog down
    # to this many candidate songs, then (slow) DTW runs only on those. Without
    # this, DTW against every sample of a ~1,200-song catalog takes ~20s/query;
    # with it, ~1s. Must be >= max_candidates and comfortably larger than any
    # plausible top-K the UI shows.
    hybrid_stage1_candidates: int = 60
    # A "confident" hybrid match needs BOTH: the model and DTW agree on the
    # top song (96-98%% precise on seen songs), AND that song's best DTW
    # distance is under a floor. The floor is route-aware because the two DTW
    # variants live on different distance scales:
    #  - FULL queries use global DTW. Recalibrated 2026-07-22 against the
    #    honest full-catalog eval (scripts/calibrate_hybrid.py): when the
    #    model and DTW agree but are WRONG, best-DTW-distance was measured at
    #    0.120 and 0.125 — both comfortably UNDER the old 0.20 floor, meaning
    #    those wrong agreements were being reported as "confident". The floor
    #    is tightened to 0.10 (safely below the observed wrong-agreement
    #    distances) at the cost of some recall on genuinely-correct matches
    #    that scored 0.10-0.36 (a real but smaller loss than reporting wrong
    #    matches as confident).
    #  - SHORT queries use subsequence DTW, which (by design) finds the best
    #    matching fragment and so scores everything lower AND separates real
    #    from unrelated only weakly (same-song ~0.02-0.05 vs diff-song
    #    ~0.06-0.12 in the full-catalog eval, heavily overlapping). Short
    #    rhythm fragments are genuinely ambiguous, so we keep a strict floor:
    #    short queries fire "confident" only on a near-exact match and
    #    otherwise return a candidate list (the intended "here are the top
    #    few" UX for short taps). 0.04 already sits below every observed
    #    wrong-agreement distance at this scale, so it's left unchanged.
    # Everything not confident still returns ranked candidates — nothing lost.
    hybrid_confident_max_distance_full: float = 0.10
    hybrid_confident_max_distance_short: float = 0.04

    cors_allow_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_allow_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


settings = Settings()
