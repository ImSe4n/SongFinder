import random

import pytest

torch = pytest.importorskip("torch")

from app.ml.rhythm_encoder import (  # noqa: E402
    LOG_RATIO_CLIP,
    RhythmEncoder,
    batch_hard_triplet_loss,
    collate_sequences,
    featurize,
    head_crop_taps,
    load_encoder,
    random_crop_taps,
    save_encoder,
)
from app.rhythm.features import iois_to_log_ratio, taps_to_iois  # noqa: E402


def test_featurize_without_jitter_matches_production_features():
    taps = [0, 200, 500, 900, 1200, 1600]
    expected = iois_to_log_ratio(taps_to_iois(taps))
    got = featurize(taps).tolist()
    assert got == pytest.approx(expected, abs=1e-6)


def test_featurize_clips_extreme_ratios():
    # 40ms then 30000ms gap -> log ratio ~6.6, must clip to LOG_RATIO_CLIP
    taps = [0, 40, 30040, 30080]
    values = featurize(taps).tolist()
    assert all(abs(v) <= LOG_RATIO_CLIP for v in values)


def test_random_crop_is_contiguous_and_respects_min():
    rng = random.Random(0)
    taps = list(range(0, 8000, 250))  # 32 taps
    for _ in range(50):
        crop = random_crop_taps(taps, rng, min_taps=8)
        assert len(crop) >= 8
        start = taps.index(crop[0])
        assert crop == taps[start : start + len(crop)]


def test_random_crop_short_input_returned_whole():
    rng = random.Random(0)
    taps = [0, 200, 400, 600]
    assert random_crop_taps(taps, rng, min_taps=8) == taps


def test_head_crop():
    taps = list(range(0, 5000, 250))
    assert head_crop_taps(taps, 12) == taps[:12]
    assert head_crop_taps(taps[:5], 12) == taps[:5]


def test_collate_padding_and_mask():
    feats = [torch.ones(3), torch.ones(7)]
    batch, mask = collate_sequences(feats)
    assert batch.shape == (2, 7) and mask.shape == (2, 7)
    assert mask[0].tolist() == [True] * 3 + [False] * 4
    assert batch[0, 3:].abs().sum() == 0


def test_encoder_handles_variable_lengths_and_normalizes():
    model = RhythmEncoder()
    feats = [featurize([0, 200, 500, 900]), featurize(list(range(0, 8000, 250)))]
    emb = model.embed(feats)
    assert emb.shape == (2, model.embed_dim)
    assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-5)


def test_triplet_loss_zero_when_classes_well_separated():
    e = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    assert float(batch_hard_triplet_loss(e, labels, margin=0.3)) == 0.0


def test_triplet_loss_positive_when_classes_collide():
    e = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    assert float(batch_hard_triplet_loss(e, labels, margin=0.3)) > 0.0


def test_training_smoke_separates_two_synthetic_rhythms():
    """A tiny end-to-end check: 30 optimization steps on two clearly distinct
    rhythm patterns must yield embeddings that retrieve the right 'song' for
    fresh jittered queries. Fast (<15s CPU), deterministic seeds."""
    torch.manual_seed(0)
    rng = random.Random(0)

    steady = [i * 400 for i in range(12)]
    longshort = []
    t = 0
    for i in range(12):
        longshort.append(t)
        t += 600 if i % 2 == 0 else 200

    def jittered(base: list[int]) -> torch.Tensor:
        return featurize(base, jitter=0.05, rng=rng)

    model = RhythmEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(30):
        feats = [jittered(steady) for _ in range(8)] + [jittered(longshort) for _ in range(8)]
        labels = torch.tensor([0] * 8 + [1] * 8)
        batch, mask = collate_sequences(feats)
        loss = batch_hard_triplet_loss(model(batch, mask), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    db = model.embed([featurize(steady), featurize(longshort)])
    queries = model.embed([jittered(steady) for _ in range(10)] + [jittered(longshort) for _ in range(10)])
    predicted = torch.cdist(queries, db).argmin(dim=1)
    truth = torch.tensor([0] * 10 + [1] * 10)
    accuracy = float((predicted == truth).float().mean())
    assert accuracy >= 0.8, f"synthetic retrieval accuracy too low: {accuracy}"


def test_save_load_roundtrip(tmp_path):
    model = RhythmEncoder()
    path = tmp_path / "encoder.pt"
    save_encoder(model, path, eval_results={"seen": {"top1": 0.5}})
    loaded, info = load_encoder(path)

    feats = [featurize([0, 200, 500, 900, 1300])]
    assert torch.allclose(model.embed(feats), loaded.embed(feats), atol=1e-6)
    assert info.eval_results["seen"]["top1"] == 0.5
    assert "NonCommercial" in info.license_note or "non-commercial" in info.license_note.lower()
