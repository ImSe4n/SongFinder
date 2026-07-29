"""Honest production-scale evaluation: how does the v2 routed system do when
the retrieval database is the ENTIRE ~1,200-song catalog, not just the 56
QBT-Extended songs?

Why this matters (from the ship/no-ship research): retrieval top-1 degrades
roughly power-law with database size, and worse for rhythm (more near-collisions
in a big pop catalog). The 56-song benchmark is a source-domain, small-database
number; this replaces the extrapolation with a measured one.

Method:
  - queries: the same held-out human tap queries from the 56 QBT songs
    (same split/seed as every other benchmark);
  - database: those 56 songs' human reference samples PLUS every OTHER catalog
    song's machine-extracted 'auto' rhythm samples as distractors. A correct
    retrieval must now beat ~1,200 songs' worth of references.

The auto-sample distractors are noisier than human references, so this is an
approximation — but it's a far more honest scale test than 56 songs, and it's
the number to trust for the ship decision. Reported for the ROUTED system
(short->subseqDTW+v2, full->globalDTW+v2) at both catalog sizes side by side.

Usage (from backend/):
    .venv/Scripts/python scripts/eval_full_catalog.py
"""

import argparse
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from train_rhythm_model import (  # noqa: E402
    SEED,
    SHORT_QUERY_TAPS,
    Sample,
    _rank_songs,
    _topk_accuracy,
    load_human_samples,
    make_split,
)

RRF_K = 60
DISTRACTOR_SAMPLES_PER_SONG = 1  # keep the DB size ~= n_songs so it's a fair scale test


def load_auto_distractors(db_path: str, exclude_song_ids: set[int]) -> list[Sample]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select rs.song_id, rs.raw_taps_ms, s.title, s.artist "
            "from rhythm_samples rs join songs s on rs.song_id = s.id "
            "where rs.source = 'auto'"
        ).fetchall()
    finally:
        con.close()

    by_song: dict[int, list[Sample]] = {}
    for song_id, taps_json, title, artist in rows:
        if song_id in exclude_song_ids:
            continue
        taps = json.loads(taps_json)
        if len(taps) >= 5:
            by_song.setdefault(song_id, []).append(Sample(song_id, taps, title, artist))

    distractors = []
    for song_id, items in by_song.items():
        distractors.extend(items[:DISTRACTOR_SAMPLES_PER_SONG])
    return distractors


def dtw_rankings(system, queries, database, short):
    from app.rhythm.features import iois_to_log_ratio, taps_to_iois
    from app.rhythm.matching import dtw_distance, subsequence_dtw_distance

    fn = dtw_distance if system == "global" else subsequence_dtw_distance
    db_vecs = [iois_to_log_ratio(taps_to_iois(s.taps_ms)) for s in database]
    db_ids = [s.song_id for s in database]
    out, started = [], time.monotonic()
    for i, q in enumerate(queries, 1):
        taps = q.taps_ms[:SHORT_QUERY_TAPS] if short else q.taps_ms
        qv = iois_to_log_ratio(taps_to_iois(taps))
        out.append(_rank_songs([fn(qv, d) for d in db_vecs], db_ids))
        if i % 50 == 0:
            print(f"    {system} DTW: {i}/{len(queries)} ({i/(time.monotonic()-started):.1f}/s)")
    return out


def v2_rankings(model, queries, database, short):
    import torch

    from app.ml.rhythm_encoder_v2 import featurize_v2

    db_emb = model.embed([featurize_v2(s.taps_ms) for s in database])
    db_ids = [s.song_id for s in database]
    cropped = [q.taps_ms[:SHORT_QUERY_TAPS] if short else q.taps_ms for q in queries]
    q_emb = model.embed([featurize_v2(t) for t in cropped])
    dist = torch.cdist(q_emb, db_emb)
    return [_rank_songs(row.tolist(), db_ids) for row in dist]


def rrf(*rankings):
    fused = []
    for per_query in zip(*rankings):
        scores = {}
        for ranking in per_query:
            for rank, sid in enumerate(ranking, 1):
                scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank)
        fused.append(sorted(scores, key=scores.get, reverse=True))
    return fused


def routed_eval(model, queries, database, short):
    truths = [q.song_id for q in queries]
    v = v2_rankings(model, queries, database, short)
    if short:
        d = dtw_rankings("subseq", queries, database, short)
    else:
        d = dtw_rankings("global", queries, database, short)
    fused = rrf(d, v)
    return _topk_accuracy(fused, truths, 1), _topk_accuracy(fused, truths, 5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="songfinder.db")
    parser.add_argument("--checkpoint", default="models/rhythm_encoder_v2.pt")
    args = parser.parse_args()

    from app.ml.rhythm_encoder_v2 import load_encoder_v2

    model, _ = load_encoder_v2(args.checkpoint)
    split = make_split(load_human_samples(args.db), random.Random(SEED))

    small_db = split.database
    small_songs = {s.song_id for s in small_db}
    distractors = load_auto_distractors(args.db, exclude_song_ids=small_songs)
    full_db = small_db + distractors
    full_songs = len({s.song_id for s in full_db})

    print(
        f"Small DB: {len(small_db)} samples / {len(small_songs)} songs (chance {1/len(small_songs):.2%})\n"
        f"Full DB:  {len(full_db)} samples / {full_songs} songs (chance {1/full_songs:.2%})\n"
        f"(+{len(distractors)} auto-sample distractor songs from the real catalog)\n"
    )

    queries_all = [("seen", split.seen_queries), ("unseen", split.unseen_queries)]
    started = time.monotonic()
    print("=== ROUTED system: 56-song DB vs full-catalog DB (top-1 / top-5) ===\n")
    print(f"{'condition':<26}{'56-song DB':<22}{'full catalog DB':<22}")
    for qname, queries in queries_all:
        for short in (False, True):
            label = f"{qname} songs, {'short' if short else 'full '} query"
            s1, s5 = routed_eval(model, queries, small_db, short)
            f1, f5 = routed_eval(model, queries, full_db, short)
            print(f"{label:<26}top1 {s1:>5.1%} top5 {s5:>5.1%}    top1 {f1:>5.1%} top5 {f5:>5.1%}")

    print(f"\nEvaluated in {time.monotonic()-started:.0f}s")
    print(
        "\nThe full-catalog column is the honest production-scale number. Expect a\n"
        "drop from the 56-song column — that gap is the database-scaling cost the\n"
        "research predicted, and it's what real users will actually experience."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
