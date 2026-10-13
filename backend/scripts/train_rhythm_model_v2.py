"""Train rhythm encoder v2 and evaluate it honestly against v1's benchmark
baseline and a DTW-1NN floor, on the identical split.

Everything comparable to the v1 run: same split code, same seed, same
retrieval database, same four eval conditions. New here:

  --loss {supcon,proto}   which non-collapsing objective to train
  --backbone {tcn,restcn,cnn}  dilated-TCN, residual dilated-TCN, or plain CNN
  --no-aug / --no-2ch     ablation switches (2-channel + augmentation are on
                          by default; turning them off isolates their effect)

Checkpoint selection uses a validation set carved from the training samples
(carve_validation), never the reported seen/unseen queries.

Prints per-epoch collapse diagnostics (alignment / uniformity / separation /
effective-rank) so we can SEE whether the collapse that killed v1 is fixed,
not just infer it from accuracy.

Usage (from backend/, uvicorn may stay up — reads are read-only):
    .venv/Scripts/python scripts/train_rhythm_model_v2.py --loss supcon --quick
    .venv/Scripts/python scripts/train_rhythm_model_v2.py --loss proto
"""

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from train_rhythm_model import (  # noqa: E402  (reuse v1's data plumbing verbatim)
    SEED,
    SHORT_QUERY_TAPS,
    Sample,
    _rank_songs,
    _topk_accuracy,
    load_human_samples,
    make_split,
)

BATCH_SONGS = 12
BATCH_SAMPLES_PER_SONG = 6  # K; proto uses n_support of these + rest as queries
PROTO_SUPPORT = 3
CROP_MIN_TAPS = 8
VAL_SAMPLES_PER_SONG = 2


def featurize_sample(model_uses_2ch, taps, rng=None, augment=False):
    from app.ml.rhythm_encoder_v2 import featurize_v2

    return featurize_v2(taps, rng=rng, augment=augment)


def eval_model_v2(model, queries, database, short: bool) -> tuple[float, float]:
    import torch

    from app.ml.rhythm_encoder_v2 import featurize_v2

    if not queries:
        return 0.0, 0.0
    db_feats = [featurize_v2(s.taps_ms) for s in database]
    db_ids = [s.song_id for s in database]
    db_emb = model.embed(db_feats)

    q_feats = [featurize_v2(q.taps_ms[:SHORT_QUERY_TAPS] if short else q.taps_ms) for q in queries]
    q_emb = model.embed(q_feats)

    dist = torch.cdist(q_emb, db_emb)
    ranked = [_rank_songs(row.tolist(), db_ids) for row in dist]
    truths = [q.song_id for q in queries]
    return _topk_accuracy(ranked, truths, 1), _topk_accuracy(ranked, truths, 5)


def eval_dtw_1nn(queries, database, short: bool) -> tuple[float, float]:
    """Non-learned floor: 1-nearest-neighbour by global DTW on log-ratios.
    If the model can't clear this, it isn't earning its complexity."""
    from app.rhythm.features import iois_to_log_ratio, taps_to_iois
    from app.rhythm.matching import dtw_distance

    def vec(taps):
        return iois_to_log_ratio(taps_to_iois(taps))

    db_vecs = [vec(s.taps_ms) for s in database]
    db_ids = [s.song_id for s in database]
    ranked, truths = [], []
    for q in queries:
        qv = vec(q.taps_ms[:SHORT_QUERY_TAPS] if short else q.taps_ms)
        ranked.append(_rank_songs([dtw_distance(qv, d) for d in db_vecs], db_ids))
        truths.append(q.song_id)
    return _topk_accuracy(ranked, truths, 1), _topk_accuracy(ranked, truths, 5)


def carve_validation(split, rng, per_song: int = VAL_SAMPLES_PER_SONG) -> list:
    """Move a few training samples per seen song into a validation set for
    checkpoint selection, so the reported seen-query numbers come from
    queries that played no part in picking the checkpoint.

    Only samples that are NOT in the retrieval database are eligible —
    make_split puts the first DB_SAMPLES_PER_SONG training samples of each
    seen song in the database too, and a validation query that is also a
    database entry would trivially match itself. Songs without spare samples
    beyond that and a minimal training pool just contribute none."""
    in_db = {id(s) for s in split.database}
    by_song = defaultdict(list)
    for s in split.train:
        by_song[s.song_id].append(s)
    val, val_ids = [], set()
    for items in by_song.values():
        spare = [s for s in items if id(s) not in in_db]
        if len(items) - per_song < BATCH_SAMPLES_PER_SONG or len(spare) < per_song:
            continue
        for s in rng.sample(spare, per_song):
            val.append(s)
            val_ids.add(id(s))
    split.train = [s for s in split.train if id(s) not in val_ids]
    return val


def train_supcon(split, val_queries, model, epochs, rng, augment):
    import torch

    from app.ml.rhythm_encoder_v2 import (
        collate_2ch,
        embedding_diagnostics,
        featurize_v2,
        random_crop_taps,
        supcon_loss,
    )

    by_song = defaultdict(list)
    for s in split.train:
        by_song[s.song_id].append(s)
    song_ids = [sid for sid, items in by_song.items() if len(items) >= 2]

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    steps = max(1, len(split.train) // (BATCH_SONGS * BATCH_SAMPLES_PER_SONG))
    best_top1, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps):
            batch_songs = rng.sample(song_ids, min(BATCH_SONGS, len(song_ids)))
            feats, labels = [], []
            for sid in batch_songs:
                pool = by_song[sid]
                for s in rng.choices(pool, k=BATCH_SAMPLES_PER_SONG):
                    taps = random_crop_taps(s.taps_ms, rng, CROP_MIN_TAPS)
                    feats.append(featurize_v2(taps, rng=rng, augment=augment))
                    labels.append(sid)
            x, mask = collate_2ch(feats)
            emb = model(x, mask)
            loss = supcon_loss(emb, torch.tensor(labels))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        sched.step()

        if epoch % 10 == 0 or epoch == epochs:
            top1, top5 = eval_model_v2(model, val_queries, split.database, short=False)
            short_top1, _ = eval_model_v2(model, val_queries, split.database, short=True)
            with torch.no_grad():
                db_emb = model.embed([featurize_v2(s.taps_ms) for s in split.database])
                diag = embedding_diagnostics(db_emb, torch.tensor([s.song_id for s in split.database]))
            print(
                f"  epoch {epoch:3d}/{epochs}  loss {total/steps:.3f}  "
                f"val top1 {top1:.1%} short-top1 {short_top1:.1%}  | sep {diag['separation']:.2f} "
                f"effrank {diag['eff_rank']:.0f} align {diag['align']:.3f} unif {diag['uniform']:.2f}"
            )
            # Model selection: blend full- and short-query top1, not full alone.
            # Real app queries are short (6-15 taps) -- that's the condition
            # that matters most and the one a full-query-only checkpoint pick
            # can silently regress on.
            blended = 0.5 * top1 + 0.5 * short_top1
            if blended > best_top1:
                best_top1, best_state = blended, {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_proto(split, val_queries, model, epochs, rng, augment):
    import torch

    from app.ml.rhythm_encoder_v2 import (
        collate_2ch,
        embedding_diagnostics,
        featurize_v2,
        prototypical_loss,
        random_crop_taps,
    )

    by_song = defaultdict(list)
    for s in split.train:
        by_song[s.song_id].append(s)
    # episodic training needs support+query per class
    song_ids = [sid for sid, items in by_song.items() if len(items) >= PROTO_SUPPORT + 2]

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    episodes = max(1, len(split.train) // (BATCH_SONGS * BATCH_SAMPLES_PER_SONG))
    best_top1, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        total, acc_sum = 0.0, 0.0
        for _ in range(episodes):
            way = rng.sample(song_ids, min(BATCH_SONGS, len(song_ids)))
            feats, labels = [], []
            for sid in way:
                pool = by_song[sid]
                chosen = rng.sample(pool, min(BATCH_SAMPLES_PER_SONG, len(pool)))
                if len(chosen) < PROTO_SUPPORT + 1:
                    chosen = chosen + rng.choices(pool, k=PROTO_SUPPORT + 1 - len(chosen))
                for s in chosen:
                    taps = random_crop_taps(s.taps_ms, rng, CROP_MIN_TAPS)
                    feats.append(featurize_v2(taps, rng=rng, augment=augment))
                    labels.append(sid)
            x, mask = collate_2ch(feats)
            emb = model(x, mask)
            loss, acc = prototypical_loss(emb, torch.tensor(labels), n_support=PROTO_SUPPORT)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            acc_sum += float(acc)
        sched.step()

        if epoch % 10 == 0 or epoch == epochs:
            top1, top5 = eval_model_v2(model, val_queries, split.database, short=False)
            short_top1, _ = eval_model_v2(model, val_queries, split.database, short=True)
            with torch.no_grad():
                db_emb = model.embed([featurize_v2(s.taps_ms) for s in split.database])
                diag = embedding_diagnostics(db_emb, torch.tensor([s.song_id for s in split.database]))
            print(
                f"  epoch {epoch:3d}/{epochs}  loss {total/episodes:.3f} epiacc {acc_sum/episodes:.1%}  "
                f"val top1 {top1:.1%} short-top1 {short_top1:.1%}  | sep {diag['separation']:.2f} "
                f"effrank {diag['eff_rank']:.0f} align {diag['align']:.3f} unif {diag['uniform']:.2f}"
            )
            # See train_supcon's identical comment: blend short+full so model
            # selection doesn't silently regress the condition real users hit.
            blended = 0.5 * top1 + 0.5 * short_top1
            if blended > best_top1:
                best_top1, best_state = blended, {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="songfinder.db")
    parser.add_argument("--loss", choices=["supcon", "proto"], default="supcon")
    parser.add_argument("--backbone", choices=["tcn", "restcn", "cnn"], default="tcn")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="models/rhythm_encoder_v2.pt")
    parser.add_argument("--tag", default="", help="label printed in the results header")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch CPU threads. The model is tiny, so per-op threading overhead dominates: "
        "measured 0.8s/epoch at 1 thread vs 27s/epoch at 4 (two runs sharing 4 cores)",
    )
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)

    from app.ml.rhythm_encoder_v2 import RhythmEncoderV2, save_encoder_v2

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    samples = load_human_samples(args.db)
    split = make_split(samples, rng)
    n_db_songs = len({s.song_id for s in split.database})
    augment = not args.no_aug
    val_queries = carve_validation(split, rng)
    print(
        f"[{args.tag or args.loss}] corpus {len(samples)} | train {len(split.train)} | "
        f"db {len(split.database)}/{n_db_songs} songs | "
        f"val {len(val_queries)} | queries {len(split.seen_queries)} seen / {len(split.unseen_queries)} unseen | "
        f"loss={args.loss} backbone={args.backbone} aug={augment}"
    )
    print(f"chance top-1 ≈ {1/n_db_songs:.1%}\n")

    epochs = 3 if args.quick else args.epochs
    if args.quick:
        split.seen_queries = split.seen_queries[:40]
        split.unseen_queries = split.unseen_queries[:20]
        val_queries = val_queries[:40]

    model = RhythmEncoderV2(backbone=args.backbone)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}\nTraining ({args.loss})...")
    started = time.monotonic()
    trainer = train_supcon if args.loss == "supcon" else train_proto
    model = trainer(split, val_queries, model, epochs, rng, augment)
    print(f"training took {time.monotonic()-started:.0f}s\n")

    conditions = [
        ("seen songs,  full query", split.seen_queries, False),
        ("seen songs,  short query", split.seen_queries, True),
        ("unseen songs, full query", split.unseen_queries, False),
        ("unseen songs, short query", split.unseen_queries, True),
    ]
    print(f"=== v2 model vs DTW-1NN floor (db {len(split.database)}/{n_db_songs}, chance {1/n_db_songs:.1%}) ===\n")
    print(f"{'condition':<27}{'DTW-1NN floor':<24}{'v2 model':<24}")
    results = {}
    for label, queries, short in conditions:
        f1, f5 = eval_dtw_1nn(queries, split.database, short)
        m1, m5 = eval_model_v2(model, queries, split.database, short)
        results[label.strip()] = {"floor": [f1, f5], "model": [m1, m5], "n": len(queries)}
        print(f"{label:<27}top1 {f1:>5.1%} top5 {f5:>5.1%}    top1 {m1:>5.1%} top5 {m5:>5.1%}   (n={len(queries)})")

    mean_top1 = sum(r["model"][0] for r in results.values()) / len(results)
    mean_top5 = sum(r["model"][1] for r in results.values()) / len(results)
    print(f"\nv2 mean: top1 {mean_top1:.1%}  top5 {mean_top5:.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_encoder_v2(model, out, meta={"loss": args.loss, "backbone": args.backbone, "aug": augment, "results": results})
    print(f"saved -> {out} (NC-derivative, gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
