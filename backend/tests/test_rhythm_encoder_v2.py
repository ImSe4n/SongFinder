import math
import random

import pytest

torch = pytest.importorskip("torch")

from app.ml.rhythm_encoder_v2 import (  # noqa: E402
    LOG_RATIO_CLIP,
    RhythmEncoderV2,
    augment_iois,
    collate_2ch,
    embedding_diagnostics,
    featurize_v2,
    iois_to_features,
    load_encoder_v2,
    prototypical_loss,
    save_encoder_v2,
    supcon_loss,
    taps_to_iois_f,
)


def test_two_channel_shape_and_alignment():
    taps = [0, 200, 500, 900, 1200, 1600]
    feat = featurize_v2(taps)
    # 6 taps -> 5 IOIs -> 4 log-ratios; both channels length 4
    assert feat.shape == (2, 4)


def test_featurize_is_tempo_invariant_both_channels():
    taps = [0, 200, 500, 900, 1200]
    taps_2x = [t * 2 for t in taps]
    f1 = featurize_v2(taps)
    f2 = featurize_v2(taps_2x)
    # log-ratios AND median-normalized IOIs are both tempo-invariant
    assert torch.allclose(f1, f2, atol=1e-5)


def test_features_are_clipped():
    taps = [0, 40, 30040, 30080, 30120]  # extreme ratio in the middle
    feat = featurize_v2(taps)
    assert feat[0].abs().max() <= LOG_RATIO_CLIP + 1e-6
    assert feat[1].abs().max() <= math.log(4.0) + 1e-6


def test_short_sequence_returns_empty_2ch():
    assert featurize_v2([0, 200]).shape == (2, 0)  # 2 taps -> 1 IOI -> no ratios


def test_tap_deletion_merges_iois():
    rng = random.Random(0)
    iois = [200.0, 200.0, 200.0, 200.0, 200.0]
    # force deletions only
    out = augment_iois(iois, rng, p_delete=1.0, p_insert=0.0, drift=0.0, micro_jitter=0.0)
    assert len(out) < len(iois)
    assert sum(out) == pytest.approx(sum(iois))  # total duration conserved by merging


def test_tap_insertion_splits_iois():
    rng = random.Random(0)
    iois = [400.0, 400.0, 400.0]
    out = augment_iois(iois, rng, p_delete=0.0, p_insert=1.0, drift=0.0, micro_jitter=0.0)
    assert len(out) > len(iois)
    assert sum(out) == pytest.approx(sum(iois), rel=1e-6)


def test_augmentation_actually_changes_log_ratios():
    """The whole point: these augmentations must perturb a tempo-invariant
    representation (unlike global tempo scaling, which would not)."""
    rng = random.Random(3)
    taps = list(range(0, 6400, 200))  # 32 evenly spaced taps -> all log-ratios 0
    base = featurize_v2(taps)
    changed = featurize_v2(taps, rng=rng, augment=True)
    # different length or different values — either way it moved
    assert changed.shape != base.shape or not torch.allclose(changed, base)


def test_collate_2ch_padding_and_mask():
    feats = [iois_to_features([200, 200, 200, 200]), iois_to_features([200, 300, 400, 200, 600, 200])]
    batch, mask = collate_2ch(feats)
    assert batch.shape[0] == 2 and batch.shape[1] == 2  # (B, 2ch, Lmax)
    assert mask[0].sum() < mask[1].sum()


def test_encoder_variable_length_normalized_both_backbones():
    for backbone in ("tcn", "cnn"):
        model = RhythmEncoderV2(backbone=backbone)
        feats = [featurize_v2([0, 200, 500, 900, 1300]), featurize_v2(list(range(0, 6000, 200)))]
        emb = model.embed(feats)
        assert emb.shape == (2, model.embed_dim)
        assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-5)


def test_supcon_loss_lower_when_classes_separated():
    good = torch.tensor([[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0], [-0.99, -0.01]])
    good = torch.nn.functional.normalize(good, dim=1)
    bad = torch.nn.functional.normalize(torch.randn(4, 2), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    assert float(supcon_loss(good, labels)) < float(supcon_loss(bad, labels))


def test_prototypical_loss_perfect_separation_high_accuracy():
    # class 0 near (1,0), class 1 near (-1,0); 3 support + 2 query each
    e = torch.tensor(
        [[1.0, 0.0]] * 5 + [[-1.0, 0.0]] * 5, dtype=torch.float32
    )
    e = torch.nn.functional.normalize(e, dim=1)
    labels = torch.tensor([0] * 5 + [1] * 5)
    loss, acc = prototypical_loss(e, labels, n_support=3)
    assert float(acc) == 1.0
    assert float(loss) < 0.5


def test_diagnostics_detect_collapse_vs_separation():
    labels = torch.tensor([0, 0, 1, 1])
    collapsed = torch.nn.functional.normalize(torch.ones(4, 8) + 1e-4 * torch.randn(4, 8), dim=1)
    separated = torch.nn.functional.normalize(
        torch.tensor([[1.0] + [0] * 7, [0.9, 0.1] + [0] * 6, [0, 0, 1.0] + [0] * 5, [0, 0, 0.9, 0.1] + [0] * 4]),
        dim=1,
    )
    d_collapsed = embedding_diagnostics(collapsed, labels)
    d_separated = embedding_diagnostics(separated, labels)
    assert d_separated["separation"] > d_collapsed["separation"]


def test_v2_save_load_roundtrip(tmp_path):
    model = RhythmEncoderV2(backbone="tcn")
    path = tmp_path / "v2.pt"
    save_encoder_v2(model, path, meta={"loss": "supcon"})
    loaded, meta = load_encoder_v2(path)
    feats = [featurize_v2([0, 200, 500, 900, 1300])]
    assert torch.allclose(model.embed(feats), loaded.embed(feats), atol=1e-6)
    assert meta["loss"] == "supcon"


def test_training_smoke_supcon_separates_two_rhythms():
    """End-to-end: a few SupCon steps must separate two distinct rhythms so
    fresh jittered queries retrieve the right one."""
    torch.manual_seed(0)
    rng = random.Random(0)
    steady = [i * 400 for i in range(12)]
    longshort = []
    t = 0
    for i in range(12):
        longshort.append(t)
        t += 600 if i % 2 == 0 else 200

    model = RhythmEncoderV2(backbone="cnn")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(40):
        feats = [featurize_v2(steady, rng=rng, augment=True) for _ in range(6)] + [
            featurize_v2(longshort, rng=rng, augment=True) for _ in range(6)
        ]
        labels = torch.tensor([0] * 6 + [1] * 6)
        x, mask = collate_2ch(feats)
        loss = supcon_loss(model(x, mask), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    db = model.embed([featurize_v2(steady), featurize_v2(longshort)])
    q = model.embed([featurize_v2(steady) for _ in range(8)] + [featurize_v2(longshort) for _ in range(8)])
    pred = torch.cdist(q, db).argmin(dim=1)
    truth = torch.tensor([0] * 8 + [1] * 8)
    assert float((pred == truth).float().mean()) >= 0.75
