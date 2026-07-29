"""Train the learned rhythm encoder and evaluate it HONESTLY against the
production DTW matcher on identical held-out data.

The whole point of this script is the comparison, not the training: the DTW
matcher (app/rhythm/matching.py) is the incumbent, and the learned model
only earns its way into the live matching path if it beats DTW on held-out
human queries — especially the conditions that matter in production:

  - SHORT queries: real app users tap ~6-15 taps; the QBT-Extended corpus
    averages ~33. Evaluated as the first 12 taps of each held-out query.
  - UNSEEN songs: SongFinder's catalog grows after training. 10 songs are
    held out of training entirely; their samples enter the retrieval
    database only at eval time. This measures whether the learned notion of
    rhythm similarity generalizes, or just memorized the training songs.

Both systems get the exact same retrieval database (capped per song so the
brute-force DTW baseline stays computable) and the exact same queries.
Machine-guessed 'auto' samples are excluded from everything — this is a
human-vs-human evaluation.

Licensing: the corpus is substantially QBT-Extended (CC BY-NC-SA,
non-commercial). The trained checkpoint is a derivative work; backend/models/
is gitignored for exactly this reason.

Usage (from the backend/ directory; uvicorn can stay up — reads are ro):
    .venv/Scripts/python scripts/train_rhythm_model.py --quick   # smoke run
    .venv/Scripts/python scripts/train_rhythm_model.py           # real run
"""

import argparse
import json
import math
import random
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 20260718
HUMAN_SOURCES = ("train", "confirm", "external_qbte")

UNSEEN_SONG_COUNT = 10
QUERIES_PER_SONG = 5  # cap, ~20% of a QBTE song's samples
DB_SAMPLES_PER_SONG = 10  # retrieval-database cap (keeps DTW baseline computable)
SHORT_QUERY_TAPS = 12

BATCH_SONGS = 8  # P in PK sampling
BATCH_SAMPLES_PER_SONG = 4  # K
TRIPLET_MARGIN = 0.3
JITTER = 0.04
CROP_MIN_TAPS = 8


@dataclass
class Sample:
    song_id: int
    taps_ms: list[int]
    title: str
    artist: str


@dataclass
class Split:
    train: list[Sample] = field(default_factory=list)
    database: list[Sample] = field(default_factory=list)
    seen_queries: list[Sample] = field(default_factory=list)
    unseen_queries: list[Sample] = field(default_factory=list)
    unseen_song_ids: set[int] = field(default_factory=set)


def load_human_samples(db_path: str) -> list[Sample]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" * len(HUMAN_SOURCES))
        rows = con.execute(
            f"select rs.song_id, rs.raw_taps_ms, s.title, s.artist "
            f"from rhythm_samples rs join songs s on rs.song_id = s.id "
            f"where rs.source in ({placeholders})",
            HUMAN_SOURCES,
        ).fetchall()
    finally:
        con.close()

    samples = []
    for song_id, taps_json, title, artist in rows:
        taps = json.loads(taps_json)
        if len(taps) >= 4:
            samples.append(Sample(song_id, taps, title, artist))
    return samples


def make_split(samples: list[Sample], rng: random.Random) -> Split:
    by_song: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_song[s.song_id].append(s)

    eligible_unseen = [sid for sid, items in by_song.items() if len(items) >= 10]
    rng.shuffle(eligible_unseen)
    unseen_ids = set(eligible_unseen[:UNSEEN_SONG_COUNT])

    split = Split(unseen_song_ids=unseen_ids)
    for song_id, items in by_song.items():
        rng.shuffle(items)
        n_queries = min(QUERIES_PER_SONG, max(1, round(0.2 * len(items)))) if len(items) >= 2 else 0
        queries, rest = items[:n_queries], items[n_queries:]
        split.database.extend(rest[:DB_SAMPLES_PER_SONG])
        if song_id in unseen_ids:
            split.unseen_queries.extend(queries)
        else:
            split.seen_queries.extend(queries)
            split.train.extend(rest)  # train on ALL non-query samples of seen songs
    return split


# ---------------------------------------------------------------------------
# Evaluation (shared protocol for DTW and the model)
# ---------------------------------------------------------------------------

def _rank_songs(distances: list[float], db_song_ids: list[int]) -> list[int]:
    best: dict[int, float] = {}
    for d, sid in zip(distances, db_song_ids):
        if sid not in best or d < best[sid]:
            best[sid] = d
    return sorted(best, key=lambda sid: best[sid])


def _topk_accuracy(ranked_lists: list[list[int]], truths: list[int], k: int) -> float:
    hits = sum(1 for ranked, truth in zip(ranked_lists, truths) if truth in ranked[:k])
    return hits / max(1, len(truths))


def eval_dtw(
    queries: list[Sample],
    database: list[Sample],
    short: bool,
    progress_label: str,
    distance_fn=None,
) -> tuple[float, float]:
    from app.rhythm.features import iois_to_log_ratio, taps_to_iois
    from app.rhythm.matching import dtw_distance

    if distance_fn is None:
        distance_fn = dtw_distance

    def vec(taps: list[int]) -> list[float]:
        return iois_to_log_ratio(taps_to_iois(taps))

    db_vectors = [vec(s.taps_ms) for s in database]
    db_song_ids = [s.song_id for s in database]

    ranked_lists, truths = [], []
    started = time.monotonic()
    for i, q in enumerate(queries, start=1):
        taps = q.taps_ms[:SHORT_QUERY_TAPS] if short else q.taps_ms
        q_vec = vec(taps)
        distances = [distance_fn(q_vec, d) for d in db_vectors]
        ranked_lists.append(_rank_songs(distances, db_song_ids))
        truths.append(q.song_id)
        if i % 50 == 0:
            rate = i / (time.monotonic() - started)
            print(f"    {progress_label}: {i}/{len(queries)} queries ({rate:.1f}/s)")
    return _topk_accuracy(ranked_lists, truths, 1), _topk_accuracy(ranked_lists, truths, 5)


def eval_model(model, queries: list[Sample], database: list[Sample], short: bool) -> tuple[float, float]:
    import torch

    from app.ml.rhythm_encoder import featurize

    db_feats = [featurize(s.taps_ms) for s in database]
    db_song_ids = [s.song_id for s in database]
    db_emb = model.embed(db_feats)

    q_feats = [
        featurize(q.taps_ms[:SHORT_QUERY_TAPS] if short else q.taps_ms) for q in queries
    ]
    q_emb = model.embed(q_feats)

    dist = torch.cdist(q_emb, db_emb)  # (Q, D)
    ranked_lists = [_rank_songs(row.tolist(), db_song_ids) for row in dist]
    truths = [q.song_id for q in queries]
    return _topk_accuracy(ranked_lists, truths, 1), _topk_accuracy(ranked_lists, truths, 5)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(split: Split, epochs: int, rng: random.Random):
    import torch

    from app.ml.rhythm_encoder import (
        RhythmEncoder,
        batch_hard_triplet_loss,
        collate_sequences,
        featurize,
        random_crop_taps,
    )

    torch.manual_seed(SEED)

    by_song: dict[int, list[Sample]] = defaultdict(list)
    for s in split.train:
        by_song[s.song_id].append(s)
    song_ids = [sid for sid, items in by_song.items() if len(items) >= 2]

    model = RhythmEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.85)], gamma=0.3
    )

    steps_per_epoch = max(1, len(split.train) // (BATCH_SONGS * BATCH_SAMPLES_PER_SONG))
    best_top1, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            batch_songs = rng.sample(song_ids, min(BATCH_SONGS, len(song_ids)))
            feats, labels = [], []
            for sid in batch_songs:
                pool = by_song[sid]
                for s in rng.choices(pool, k=BATCH_SAMPLES_PER_SONG):
                    taps = random_crop_taps(s.taps_ms, rng, min_taps=CROP_MIN_TAPS)
                    feats.append(featurize(taps, jitter=JITTER, rng=rng))
                    labels.append(sid)
            batch, mask = collate_sequences(feats)
            embeddings = model(batch, mask)
            loss = batch_hard_triplet_loss(
                embeddings, torch.tensor(labels), margin=TRIPLET_MARGIN
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            top1, top5 = eval_model(model, split.seen_queries, split.database, short=False)
            print(
                f"  epoch {epoch:3d}/{epochs}  loss {epoch_loss / steps_per_epoch:.4f}  "
                f"seen-full top1 {top1:.1%} top5 {top5:.1%}"
            )
            if top1 > best_top1:
                best_top1 = top1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="songfinder.db", help="Path to the SongFinder database")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--quick", action="store_true", help="Smoke run: 3 epochs, subsampled DTW baseline"
    )
    parser.add_argument("--out", default="models/rhythm_encoder.pt")
    args = parser.parse_args()

    rng = random.Random(SEED)
    samples = load_human_samples(args.db)
    if len(samples) < 100:
        print(f"Only {len(samples)} human samples — not enough to train meaningfully.")
        return 1

    split = make_split(samples, rng)
    n_songs_db = len({s.song_id for s in split.database})
    print(
        f"Corpus: {len(samples)} human samples | train {len(split.train)} | "
        f"database {len(split.database)} across {n_songs_db} songs | "
        f"queries: {len(split.seen_queries)} seen-song, {len(split.unseen_queries)} unseen-song "
        f"({len(split.unseen_song_ids)} songs fully held out of training)"
    )
    print(f"Chance top-1 ≈ {1 / n_songs_db:.1%}\n")

    epochs = 3 if args.quick else args.epochs
    if args.quick:
        split.seen_queries = split.seen_queries[:40]
        split.unseen_queries = split.unseen_queries[:20]

    print("Training encoder...")
    started = time.monotonic()
    model = train_model(split, epochs, rng)
    print(f"Training took {time.monotonic() - started:.0f}s\n")

    conditions = [
        ("seen songs,  full query ", split.seen_queries, False),
        ("seen songs,  short query", split.seen_queries, True),
        ("unseen songs, full query ", split.unseen_queries, False),
        ("unseen songs, short query", split.unseen_queries, True),
    ]

    results: dict[str, dict] = {}
    print("Evaluating (identical database + queries for both systems)...")
    for label, queries, short in conditions:
        m_top1, m_top5 = eval_model(model, queries, split.database, short)
        d_top1, d_top5 = eval_dtw(queries, split.database, short, label.strip())
        results[label.strip()] = {
            "dtw": {"top1": d_top1, "top5": d_top5},
            "model": {"top1": m_top1, "top5": m_top5},
            "n_queries": len(queries),
        }

    print(
        f"\n=== Retrieval accuracy — database: {len(split.database)} samples / "
        f"{n_songs_db} songs (chance top-1 ≈ {1 / n_songs_db:.1%}) ===\n"
    )
    print(f"{'condition':<28}{'DTW (production)':<24}{'Learned encoder':<24}")
    for label, r in results.items():
        d, m = r["dtw"], r["model"]
        print(
            f"{label:<28}"
            f"top1 {d['top1']:>5.1%}  top5 {d['top5']:>5.1%}    "
            f"top1 {m['top1']:>5.1%}  top5 {m['top5']:>5.1%}    "
            f"(n={r['n_queries']})"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from app.ml.rhythm_encoder import save_encoder

    save_encoder(model, out_path, eval_results=results)
    print(f"\nCheckpoint saved to {out_path} (NC-license derivative — gitignored, do not ship).")

    model_wins_short = all(
        results[c]["model"]["top1"] >= results[c]["dtw"]["top1"]
        for c in results
        if "short" in c
    )
    print(
        "\nVerdict: "
        + (
            "the learned encoder matches or beats DTW on the short-query conditions "
            "that mirror real app usage — worth wiring in as an option."
            if model_wins_short
            else "DTW still wins on short queries — keep DTW in production; retrain as "
            "more real app-native samples accumulate."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
