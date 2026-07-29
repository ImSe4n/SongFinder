"""Definitive matcher benchmark: three base systems and their rank-fusion
combinations, on identical held-out human queries and an identical retrieval
database.

Base systems:
  - global DTW        — production (app/rhythm/matching.py: dtw_distance)
  - subsequence DTW   — classical fix for short-query-vs-long-reference
  - learned encoder   — trained by scripts/train_rhythm_model.py

Fusions (reciprocal-rank fusion, scale-free):
  - global DTW + encoder
  - subsequence DTW + encoder
  - all three

Each base system's per-query song rankings are computed once; every fusion
is derived from those, so all numbers come from one run over one split (the
training script's exact split — same seed, same code).

Usage (from the backend/ directory, after train_rhythm_model.py has run):
    .venv/Scripts/python scripts/eval_matchers.py
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

RRF_K = 60  # standard reciprocal-rank-fusion constant

BASE_SYSTEMS = ("global DTW", "subsequence DTW", "learned encoder")
FUSIONS = {
    "globalDTW+encoder": ("global DTW", "learned encoder"),
    "subseqDTW+encoder": ("subsequence DTW", "learned encoder"),
    "all three fused": ("global DTW", "subsequence DTW", "learned encoder"),
}


def compute_ranked_lists(
    system: str, model, queries: list[Sample], database: list[Sample], short: bool
) -> list[list[int]]:
    """Per-query ranked song-id lists for one base system."""
    import torch

    from app.ml.rhythm_encoder import featurize
    from app.rhythm.features import iois_to_log_ratio, taps_to_iois
    from app.rhythm.matching import dtw_distance, subsequence_dtw_distance

    def vec(taps: list[int]) -> list[float]:
        return iois_to_log_ratio(taps_to_iois(taps))

    db_song_ids = [s.song_id for s in database]
    cropped = [q.taps_ms[:SHORT_QUERY_TAPS] if short else q.taps_ms for q in queries]

    if system == "learned encoder":
        db_emb = model.embed([featurize(s.taps_ms) for s in database])
        q_emb = model.embed([featurize(taps) for taps in cropped])
        dist = torch.cdist(q_emb, db_emb)
        return [_rank_songs(row.tolist(), db_song_ids) for row in dist]

    distance_fn = dtw_distance if system == "global DTW" else subsequence_dtw_distance
    db_vectors = [vec(s.taps_ms) for s in database]
    ranked, started = [], time.monotonic()
    for i, taps in enumerate(cropped, start=1):
        q_vec = vec(taps)
        ranked.append(_rank_songs([distance_fn(q_vec, d) for d in db_vectors], db_song_ids))
        if i % 100 == 0:
            print(f"    {system}: {i}/{len(cropped)} ({i / (time.monotonic() - started):.1f}/s)")
    return ranked


def rrf_fuse(rankings_per_system: list[list[list[int]]]) -> list[list[int]]:
    """Fuse per-query rankings from several systems via reciprocal-rank fusion."""
    fused = []
    for per_query in zip(*rankings_per_system):
        scores: dict[int, float] = {}
        for ranking in per_query:
            for rank, sid in enumerate(ranking, start=1):
                scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank)
        fused.append(sorted(scores, key=scores.get, reverse=True))
    return fused


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="songfinder.db")
    parser.add_argument("--checkpoint", default="models/rhythm_encoder.pt")
    args = parser.parse_args()

    from app.ml.rhythm_encoder import load_encoder

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"No checkpoint at {checkpoint} — run scripts/train_rhythm_model.py first.")
        return 1
    model, _ = load_encoder(checkpoint)

    samples = load_human_samples(args.db)
    split = make_split(samples, random.Random(SEED))
    n_songs_db = len({s.song_id for s in split.database})
    print(
        f"Database: {len(split.database)} samples / {n_songs_db} songs "
        f"(chance top-1 ≈ {1 / n_songs_db:.1%}) | queries: "
        f"{len(split.seen_queries)} seen-song, {len(split.unseen_queries)} unseen-song\n"
    )

    conditions = [
        ("seen songs,  full query", split.seen_queries, False),
        ("seen songs,  short query", split.seen_queries, True),
        ("unseen songs, full query", split.unseen_queries, False),
        ("unseen songs, short query", split.unseen_queries, True),
    ]

    started = time.monotonic()
    # accuracy[system][condition] = (top1, top5)
    accuracy: dict[str, dict[str, tuple[float, float]]] = {
        name: {} for name in list(BASE_SYSTEMS) + list(FUSIONS)
    }
    for cond_label, queries, short in conditions:
        truths = [q.song_id for q in queries]
        base_rankings: dict[str, list[list[int]]] = {}
        for system in BASE_SYSTEMS:
            ranked = compute_ranked_lists(system, model, queries, split.database, short)
            base_rankings[system] = ranked
            accuracy[system][cond_label] = (
                _topk_accuracy(ranked, truths, 1),
                _topk_accuracy(ranked, truths, 5),
            )
        for fusion_name, members in FUSIONS.items():
            fused = rrf_fuse([base_rankings[m] for m in members])
            accuracy[fusion_name][cond_label] = (
                _topk_accuracy(fused, truths, 1),
                _topk_accuracy(fused, truths, 5),
            )
        print(f"  done: {cond_label}")
    print(f"\nEvaluation took {time.monotonic() - started:.0f}s\n")

    cond_labels = [c for c, _, _ in conditions]
    print("=== Retrieval accuracy, top-1 / top-5 ===\n")
    header = f"{'system':<22}" + "".join(f"{c.replace(' songs, ', '/'):<22}" for c in cond_labels)
    print(header + "mean top-1")
    for system, per_cond in accuracy.items():
        row = f"{system:<22}"
        for cond in cond_labels:
            top1, top5 = per_cond[cond]
            row += f"{top1:>5.1%}/{top5:>5.1%}         "
        mean_top1 = sum(per_cond[c][0] for c in cond_labels) / len(cond_labels)
        print(row + f"{mean_top1:>6.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
