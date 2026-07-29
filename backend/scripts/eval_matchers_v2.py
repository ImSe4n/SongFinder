"""Final benchmark for the v2 encoder: standalone and in rank-fusion with the
two DTW variants, against the same split and the same bar the v1 work set —
the routed-fusion baseline of ~50%% top-1 / 76%% top-5 (mean over the four
conditions of: short-query -> subseqDTW+encoder, full-query -> globalDTW+encoder).

Systems evaluated:
  global DTW, subsequence DTW  (unchanged production/classical baselines)
  v2 encoder                    (SupCon/TCN, the ablation winner)
  globalDTW + v2                (best for full queries in v1 pattern)
  subseqDTW + v2                (best for short queries in v1 pattern)
  all three fused
  ROUTED (short->subseq+v2, full->global+v2)   <- the production-shaped number

Usage (from backend/):
    .venv/Scripts/python scripts/eval_matchers_v2.py
"""

import argparse
import random
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


def dtw_rankings(system, queries, database, short):
    from app.rhythm.features import iois_to_log_ratio, taps_to_iois
    from app.rhythm.matching import dtw_distance, subsequence_dtw_distance

    fn = dtw_distance if system == "global DTW" else subsequence_dtw_distance
    db_vecs = [iois_to_log_ratio(taps_to_iois(s.taps_ms)) for s in database]
    db_ids = [s.song_id for s in database]
    out, started = [], time.monotonic()
    for i, q in enumerate(queries, 1):
        taps = q.taps_ms[:SHORT_QUERY_TAPS] if short else q.taps_ms
        qv = iois_to_log_ratio(taps_to_iois(taps))
        out.append(_rank_songs([fn(qv, d) for d in db_vecs], db_ids))
        if i % 100 == 0:
            print(f"    {system}: {i}/{len(queries)} ({i/(time.monotonic()-started):.1f}/s)")
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


def rrf(*rankings_per_system):
    fused = []
    for per_query in zip(*rankings_per_system):
        scores = {}
        for ranking in per_query:
            for rank, sid in enumerate(ranking, 1):
                scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank)
        fused.append(sorted(scores, key=scores.get, reverse=True))
    return fused


def acc(ranked, truths):
    return _topk_accuracy(ranked, truths, 1), _topk_accuracy(ranked, truths, 5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="songfinder.db")
    parser.add_argument("--checkpoint", default="models/rhythm_encoder_v2.pt")
    args = parser.parse_args()

    from app.ml.rhythm_encoder_v2 import load_encoder_v2

    cp = Path(args.checkpoint)
    if not cp.is_file():
        print(f"No v2 checkpoint at {cp}.")
        return 1
    model, meta = load_encoder_v2(cp)
    print(f"v2 checkpoint: {meta}\n")

    split = make_split(load_human_samples(args.db), random.Random(SEED))
    n_db = len({s.song_id for s in split.database})
    print(f"Database {len(split.database)} samples / {n_db} songs (chance {1/n_db:.1%})\n")

    conditions = [
        ("seen songs,  full query", split.seen_queries, False),
        ("seen songs,  short query", split.seen_queries, True),
        ("unseen songs, full query", split.unseen_queries, False),
        ("unseen songs, short query", split.unseen_queries, True),
    ]

    systems = [
        "global DTW",
        "subsequence DTW",
        "v2 encoder",
        "globalDTW+v2",
        "subseqDTW+v2",
        "all fused",
        "ROUTED",
    ]
    results = {s: {} for s in systems}
    started = time.monotonic()

    for cond, queries, short in conditions:
        truths = [q.song_id for q in queries]
        g = dtw_rankings("global DTW", queries, split.database, short)
        s = dtw_rankings("subsequence DTW", queries, split.database, short)
        v = v2_rankings(model, queries, split.database, short)

        results["global DTW"][cond] = acc(g, truths)
        results["subsequence DTW"][cond] = acc(s, truths)
        results["v2 encoder"][cond] = acc(v, truths)
        results["globalDTW+v2"][cond] = acc(rrf(g, v), truths)
        results["subseqDTW+v2"][cond] = acc(rrf(s, v), truths)
        results["all fused"][cond] = acc(rrf(g, s, v), truths)
        # ROUTED: short queries use subseq+v2, full queries use global+v2
        routed = rrf(s, v) if short else rrf(g, v)
        results["ROUTED"][cond] = acc(routed, truths)
        print(f"  done: {cond}")

    print(f"\nEvaluated in {time.monotonic()-started:.0f}s\n")
    cond_labels = [c for c, _, _ in conditions]
    print("=== top-1 / top-5 ===\n")
    print(f"{'system':<20}" + "".join(f"{c.replace(' songs,',''):<21}" for c in cond_labels) + "mean t1")
    for sysname in systems:
        row = f"{sysname:<20}"
        for c in cond_labels:
            t1, t5 = results[sysname][c]
            row += f"{t1:>5.1%}/{t5:>5.1%}       "
        mean1 = sum(results[sysname][c][0] for c in cond_labels) / len(cond_labels)
        print(row + f"{mean1:>6.1%}")

    routed_mean1 = sum(results["ROUTED"][c][0] for c in cond_labels) / len(cond_labels)
    routed_mean5 = sum(results["ROUTED"][c][1] for c in cond_labels) / len(cond_labels)
    print(f"\nROUTED (production-shaped): mean top-1 {routed_mean1:.1%}, mean top-5 {routed_mean5:.1%}")
    print("Baseline to beat (v1 routed fusion): ~50.0% top-1 / ~76.0% top-5")
    verdict = "BEATS" if routed_mean1 > 0.50 else "does NOT beat"
    print(f"Verdict: v2 routed {verdict} the v1 baseline on mean top-1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
