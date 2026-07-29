"""Hybrid rhythm matching: the learned encoder + DTW, fused and routed by
query length. This is the production form of the modeling work benchmarked in
scripts/eval_matchers_v2.py and scripts/eval_full_catalog.py.

Design, and the evidence behind each choice:

* Routed fusion. Short queries (<= hybrid_short_query_taps) route through
  subsequence-DTW; longer queries through global DTW; the learned encoder is
  fused on top in both cases via reciprocal-rank fusion. This routed system
  measured best across every condition (mean top-1 55.5% at 56-song scale,
  and full-query accuracy holds ~78% even against the full ~1,200-song
  catalog).

* Agreement-based confidence, NOT a fused-score threshold. Measured directly:
  RRF scores barely separate correct from wrong matches (correct top-score
  median 0.033 vs wrong 0.031), so a threshold on them is unreliable. But
  when the encoder and DTW *independently* agree on the top song, that song
  is correct 96-98% of the time on seen songs. So: agree -> confident;
  disagree -> ambiguous (return the candidate list, let the user pick, which
  matches the "surface top-5" UX the scaling analysis recommended).

* Graceful fallback. If the model is disabled or the checkpoint can't load,
  callers fall back to the pure-DTW path. The model is strictly additive.

* Two-stage retrieval (this is what makes it shippable at catalog scale).
  Running DTW against every reference sample of a ~1,200-song catalog takes
  ~20s/query — unusable. So the fast model embedding is a first-stage FILTER:
  embed the query (~140ms), take the top `hybrid_stage1_candidates` songs by
  embedding distance, and run the expensive DTW only on those. Measured: the
  true song lands in the stage-1 top-50 essentially always, and total query
  time drops from ~22s to ~1s. The catalog embedding matrix is built once and
  cached, invalidated per-song when a song's sample set changes.
"""

import threading

from app.config import settings

_model = None
_model_load_attempted = False
_model_lock = threading.Lock()

# Per-song embedding cache: {song_id: (embedding_tensor (n_samples, D), signature)}
_emb_cache: dict[int, object] = {}
_emb_cache_lock = threading.Lock()


class HybridUnavailable(Exception):
    pass


def _load_model():
    """Lazy, once. Returns the model or raises HybridUnavailable; the failure
    is cached so we don't retry a missing/broken checkpoint every request."""
    global _model, _model_load_attempted
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if _model_load_attempted:
            raise HybridUnavailable("model previously failed to load")
        _model_load_attempted = True
        try:
            from pathlib import Path

            from app.ml.rhythm_encoder_v2 import load_encoder_v2

            path = Path(settings.rhythm_model_path)
            if not path.is_file():
                raise HybridUnavailable(f"checkpoint not found at {path}")
            model, _meta = load_encoder_v2(path)
            _model = model
            return _model
        except HybridUnavailable:
            raise
        except Exception as exc:  # torch missing, corrupt checkpoint, ...
            raise HybridUnavailable(f"could not load rhythm model: {exc}") from exc


def hybrid_available() -> bool:
    if not settings.hybrid_matching_enabled:
        return False
    try:
        _load_model()
        return True
    except HybridUnavailable:
        return False


def _sample_signature(taps_lists: list[list[int]]) -> tuple:
    """Cheap identity for a song's reference set — invalidates the cached
    embedding when the song gains/loses samples (e.g. a new confirm)."""
    return (len(taps_lists), tuple(len(t) for t in taps_lists))


def _song_embedding(model, song_id: int, taps_lists: list[list[int]]):
    """Best (min-distance-friendly) reference embedding for a song: we embed
    every reference sample and keep them all, matching against the closest.
    Cached per song, keyed by the sample signature."""
    import torch

    from app.ml.rhythm_encoder_v2 import featurize_v2

    sig = _sample_signature(taps_lists)
    with _emb_cache_lock:
        cached = _emb_cache.get(song_id)
        if cached is not None and cached[1] == sig:
            return cached[0]

    feats = [featurize_v2(t) for t in taps_lists if len(t) >= 3]
    if not feats:
        return None
    emb = model.embed(feats)  # (n_samples, D)
    with _emb_cache_lock:
        _emb_cache[song_id] = (emb, sig)
    return emb


def _rank_songs_from_distances(distances, song_ids):
    best: dict[int, float] = {}
    for d, sid in zip(distances, song_ids):
        if sid not in best or d < best[sid]:
            best[sid] = d
    return sorted(best, key=lambda sid: best[sid])


def rank_hybrid(taps_ms: list[int], songs):
    """`songs`: iterable of ORM Song objects (with .id and .samples). Returns
    (status, ranked_song_ids, per_song_confidence, model_top, dtw_top).

    ranked_song_ids is the fused ranking (best first). Raises HybridUnavailable
    if the model can't be used — callers must catch and fall back to DTW.
    """
    import torch

    from app.ml.rhythm_encoder_v2 import featurize_v2
    from app.rhythm.features import iois_to_log_ratio, taps_to_iois
    from app.rhythm.matching import dtw_distance, subsequence_dtw_distance

    model = _load_model()
    short = len(taps_ms) <= settings.hybrid_short_query_taps
    rrf_k = settings.hybrid_rrf_k

    per_song_taps: dict[int, list[list[int]]] = {}
    for song in songs:
        taps_lists = [s.raw_taps_ms for s in song.samples if len(s.raw_taps_ms) >= 3]
        if taps_lists:
            per_song_taps[song.id] = taps_lists

    if not per_song_taps:
        return "no_match", [], {}, None, None

    # === Stage 1: fast model filter over the whole catalog ===
    q_emb = model.embed([featurize_v2(taps_ms)])  # (1, D)
    model_dists, model_ids = [], []
    for sid, taps_lists in per_song_taps.items():
        song_emb = _song_embedding(model, sid, taps_lists)
        if song_emb is None:
            continue
        model_dists.append(torch.cdist(q_emb, song_emb)[0].min().item())
        model_ids.append(sid)
    model_ranking = _rank_songs_from_distances(model_dists, model_ids)

    # DTW only runs on the stage-1 survivors (plus, defensively, nothing else).
    stage1 = set(model_ranking[: settings.hybrid_stage1_candidates])

    # === Stage 2: expensive DTW, only on the shortlist ===
    dtw_fn = subsequence_dtw_distance if short else dtw_distance
    query_r = iois_to_log_ratio(taps_to_iois(taps_ms))
    dtw_pairs, dtw_ids = [], []
    best_dtw: dict[int, float] = {}
    for sid in stage1:
        for taps in per_song_taps[sid]:
            ref_r = iois_to_log_ratio(taps_to_iois(taps))
            d = dtw_fn(query_r, ref_r)
            dtw_pairs.append(d)
            dtw_ids.append(sid)
            if sid not in best_dtw or d < best_dtw[sid]:
                best_dtw[sid] = d
    dtw_ranking = _rank_songs_from_distances(dtw_pairs, dtw_ids)

    # === Reciprocal-rank fusion (both legs restricted to the same shortlist so
    # the model's long tail past stage-1 can't dominate the fused order) ===
    model_ranking_top = [sid for sid in model_ranking if sid in stage1]
    scores: dict[int, float] = {}
    for ranking in (dtw_ranking, model_ranking_top):
        for rank, sid in enumerate(ranking, start=1):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (rrf_k + rank)
    fused = sorted(scores, key=lambda sid: scores[sid], reverse=True)

    # --- agreement-based decision ---
    # Confident requires BOTH: the two legs agree on the top song, AND that
    # song is a genuinely good DTW match in absolute terms. The distance floor
    # stops a trivial agreement (e.g. a tiny catalog with one song, or an
    # unrelated rhythm that happens to top a weak field) from being confident.
    dtw_top = dtw_ranking[0] if dtw_ranking else None
    model_top = model_ranking[0] if model_ranking else None
    agree = dtw_top is not None and dtw_top == model_top
    max_dist = (
        settings.hybrid_confident_max_distance_short
        if short
        else settings.hybrid_confident_max_distance_full
    )
    good_dtw = agree and best_dtw.get(dtw_top, float("inf")) <= max_dist
    if not fused:
        status = "no_match"
    elif agree and good_dtw:
        status = "confident"
    else:
        status = "ambiguous"

    # Per-song "confidence" for display: normalize fused score to 0-100 by the
    # top score, purely cosmetic (the real confidence signal is the status).
    top_score = scores[fused[0]] if fused else 1.0
    confidence = {sid: round(100.0 * scores[sid] / top_score) for sid in fused}

    return status, fused, confidence, model_top, dtw_top


def warm_cache(session_factory) -> None:
    """Pre-load the model and pre-embed the catalog so the FIRST real query
    doesn't eat the ~15s one-time cost. Best-effort; safe to call in a
    background thread at startup. Never raises."""
    if not settings.hybrid_matching_enabled:
        return
    try:
        model = _load_model()
    except HybridUnavailable:
        return
    try:
        from app.services.catalog import list_matchable_songs

        with session_factory() as db:
            for song in list_matchable_songs(db):
                taps_lists = [s.raw_taps_ms for s in song.samples if len(s.raw_taps_ms) >= 3]
                if taps_lists:
                    _song_embedding(model, song.id, taps_lists)
    except Exception:
        pass  # a warmup failure must never take the server down


def clear_caches() -> None:
    """Test/maintenance hook: drop the model and embedding caches."""
    global _model, _model_load_attempted
    with _model_lock:
        _model = None
        _model_load_attempted = False
    with _emb_cache_lock:
        _emb_cache.clear()
