"""Rhythm encoder v2 — the "full stack" of research-backed improvements over
app/ml/rhythm_encoder.py, kept as a separate module so v1's trained
checkpoint and its honest benchmark stay intact as the thing to beat.

Independently-motivated changes, each traceable to a specific finding:

* Diagnosed failure (empirically): v1's embeddings collapsed — within-song
  and between-song distances both ~0.001, separation ratio ~1.3, loss pinned
  at the triplet margin. Not convergence; a degenerate minimum.

* Two-channel input: log-ratios (differential, tempo-invariant) PLUS
  median-normalized IOIs (absolute proportion — "this gap is ~2x the beat" —
  which the differential channel throws away, and which compounds tap-count
  error). Both channels are tempo-invariant.

* Augmentation that actually perturbs a tempo-invariant representation:
  - tap insertion/deletion (splits/merges IOIs — the realistic human error
    that global-tempo warping, a mathematical no-op here, does NOT model);
  - local tempo drift (a smooth accel/decel envelope — unlike global tempo,
    this changes consecutive ratios);
  - additive per-onset microtiming noise.

* Losses that don't collapse: SupCon (Khosla 2020) and Prototypical/episodic
  (Snell 2017). The latter matches our actual evaluation — generalizing to
  songs unseen in training, enrolled from a few reference taps.

* Attention pooling instead of mean+max; optional dilated-TCN backbone for a
  full-sequence receptive field without RNN overfit risk at ~3k samples.

Training-data provenance is unchanged from v1: mostly QBT-Extended
(CC BY-NC-SA, non-commercial). Checkpoints are derivative works — keep them
out of git (backend/models/ is ignored).
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_RATIO_CLIP = 2.5
NORM_IOI_CLIP = 4.0  # median-normalized IOIs clipped to [1/4, 4]x the median
EMBED_DIM = 64
MIN_IOI_MS = 40


# ---------------------------------------------------------------------------
# Augmentation — operates on IOIs (ms), before any tempo-invariant transform
# ---------------------------------------------------------------------------

def augment_iois(
    iois: list[float],
    rng: random.Random,
    p_delete: float = 0.12,
    p_insert: float = 0.12,
    drift: float = 0.10,
    micro_jitter: float = 0.05,
) -> list[float]:
    """Realistic human tap perturbations. All of these actually move a
    tempo-invariant log-ratio sequence (global tempo scaling would not).

    Edit counts are capped relative to sequence length rather than applying
    p_delete/p_insert independently per IOI. On a short crop (real app
    queries: ~6-15 taps, 5-14 IOIs) an independent per-IOI roll compounds —
    empirically it moved tap count by +/-30-40% and produced merged/split
    IOIs unlike anything a human actually taps (e.g. two ~500ms taps merging
    into ~1600ms). That's what made augmentation measurably HURT short-query
    accuracy (32.2% -> 8.7% top-1 in the aug on/off ablation). Capping the
    total edits to roughly p_delete/p_insert of the sequence length keeps the
    perturbation proportional at every length instead of compounding on short
    ones.
    """
    out = list(iois)
    max_deletes = max(0, round(p_delete * len(out)))
    max_inserts = max(0, round(p_insert * len(out)))

    # tap deletion: drop a tap => merge two adjacent IOIs into their sum
    deletes_done = 0
    i = 0
    while i < len(out) - 1 and deletes_done < max_deletes:
        if len(out) > 3 and rng.random() < 0.5:
            out[i] = out[i] + out[i + 1]
            del out[i + 1]
            deletes_done += 1
        else:
            i += 1

    # tap insertion: add a tap => split one IOI into two (roughly in half)
    inserts_done = 0
    i = 0
    while i < len(out) and inserts_done < max_inserts:
        if rng.random() < 0.5:
            frac = rng.uniform(0.35, 0.65)
            first = out[i] * frac
            second = out[i] - first
            out[i] = first
            out.insert(i + 1, second)
            inserts_done += 1
            i += 2
        else:
            i += 1

    if not out:
        return out

    # local tempo drift: smooth multiplicative accel/decel envelope
    n = len(out)
    phase = rng.uniform(0, 2 * math.pi)
    out = [
        v * (1.0 + drift * math.sin(2 * math.pi * (k / max(1, n)) + phase))
        for k, v in enumerate(out)
    ]

    # additive per-onset microtiming noise (as a fraction of each IOI)
    out = [max(1.0, v * (1.0 + rng.uniform(-micro_jitter, micro_jitter))) for v in out]
    return out


# ---------------------------------------------------------------------------
# Featurization — 2 channels, both tempo-invariant
# ---------------------------------------------------------------------------

def taps_to_iois_f(taps_ms: list[int]) -> list[float]:
    return [float(taps_ms[i + 1] - taps_ms[i]) for i in range(len(taps_ms) - 1)]


def iois_to_features(iois: list[float]) -> torch.Tensor:
    """Returns a (2, L) tensor: channel 0 = clipped log-ratios (length L),
    channel 1 = clipped log median-normalized IOIs aligned to the same L.

    Log-ratios have length len(iois)-1; we align the normalized-IOI channel to
    the *later* IOI of each ratio pair so both channels describe the same
    transitions.
    """
    clamped = [max(v, float(MIN_IOI_MS)) for v in iois]
    if len(clamped) < 2:
        return torch.zeros(2, 0, dtype=torch.float32)

    log_ratios = [math.log(clamped[i + 1] / clamped[i]) for i in range(len(clamped) - 1)]

    median = float(np.median(clamped))
    if median <= 0:
        median = 1.0
    norm = [math.log(v / median) for v in clamped]  # tempo-invariant (median cancels scale)
    norm_aligned = norm[1:]  # align to the second IOI of each ratio pair -> length L

    lr = np.clip(np.asarray(log_ratios, dtype=np.float32), -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    ni = np.clip(np.asarray(norm_aligned, dtype=np.float32), -math.log(NORM_IOI_CLIP), math.log(NORM_IOI_CLIP))
    return torch.from_numpy(np.stack([lr, ni], axis=0))


def featurize_v2(
    taps_ms: list[int],
    rng: random.Random | None = None,
    augment: bool = False,
) -> torch.Tensor:
    iois = taps_to_iois_f(taps_ms)
    if augment and rng is not None:
        iois = augment_iois(iois, rng)
    return iois_to_features(iois)


def random_crop_taps(taps_ms: list[int], rng: random.Random, min_taps: int = 8) -> list[int]:
    n = len(taps_ms)
    if n <= min_taps:
        return taps_ms
    length = rng.randint(min_taps, n)
    start = rng.randint(0, n - length)
    return taps_ms[start : start + length]


def collate_2ch(features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (2, L) tensors into (B, 2, Lmax) + (B, Lmax) bool mask."""
    max_len = max(int(f.shape[1]) for f in features)
    batch = torch.zeros(len(features), 2, max_len, dtype=torch.float32)
    mask = torch.zeros(len(features), max_len, dtype=torch.bool)
    for i, f in enumerate(features):
        n = int(f.shape[1])
        batch[i, :, :n] = f
        mask[i, :n] = True
    return batch, mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Masked additive attention pooling — learns to weight the salient part
    of the tapped hook instead of averaging over crop/padding noise."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B, C, L) -> (B, L, C)
        hc = h.transpose(1, 2)
        scores = self.score(hc).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(1)  # (B, 1, L)
        return (h * weights).sum(dim=2)  # (B, C)


class RhythmEncoderV2(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, backbone: str = "tcn"):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone_kind = backbone

        if backbone == "tcn":
            # dilated conv stack: receptive field covers the whole sequence
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(2, 48, kernel_size=3, padding=1, dilation=1),
                    nn.Conv1d(48, 64, kernel_size=3, padding=2, dilation=2),
                    nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
                    nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=8),
                ]
            )
        else:  # "cnn" — plain stack, closer to v1 but 2-channel input
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(2, 32, kernel_size=3, padding=1),
                    nn.Conv1d(32, 64, kernel_size=3, padding=1),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                ]
            )
        self.act = nn.ReLU()
        self.attn = AttentionPool(64)
        self.mean_max_gate = nn.Parameter(torch.zeros(1))  # blend attn with mean/max
        self.proj = nn.Linear(64 * 2, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(1)  # (B, 1, L)
        h = x
        for conv in self.convs:
            h = self.act(conv(h))
            h = h.masked_fill(~m, 0.0)

        attn_pool = self.attn(h, mask)
        valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pool = h.masked_fill(~m, 0.0).sum(dim=2) / valid
        z = self.proj(torch.cat([attn_pool, mean_pool], dim=1))
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def embed(self, features: list[torch.Tensor]) -> torch.Tensor:
        self.eval()
        batch, mask = collate_2ch(features)
        return self.forward(batch, mask)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020). Uses ALL same-class
    pairs as positives (our PK batches give K-1 each), smoothly — no hardest
    mining, no margin to collapse onto."""
    device = embeddings.device
    sim = embeddings @ embeddings.t() / temperature
    # numerical stability
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=device)
    positives = same & ~self_mask

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_counts = positives.sum(dim=1)
    valid = pos_counts > 0
    mean_log_prob_pos = (log_prob * positives).sum(dim=1)[valid] / pos_counts[valid]
    if mean_log_prob_pos.numel() == 0:
        return embeddings.sum() * 0.0
    return -mean_log_prob_pos.mean()


def prototypical_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, n_support: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prototypical-network episodic loss (Snell et al. 2017). For each class,
    the first `n_support` samples form a prototype (mean); the rest are queries
    classified by nearest prototype. Returns (loss, accuracy).

    Assumes the batch is laid out as consecutive blocks per class with at
    least n_support+1 samples each (the episodic sampler guarantees this)."""
    classes = torch.unique(labels)
    prototypes = []
    query_embs = []
    query_targets = []
    for ci, c in enumerate(classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        support = idx[:n_support]
        query = idx[n_support:]
        prototypes.append(embeddings[support].mean(dim=0))
        query_embs.append(embeddings[query])
        query_targets.append(torch.full((len(query),), ci, device=embeddings.device))
    protos = torch.stack(prototypes)  # (C, D)
    q = torch.cat(query_embs)  # (Q, D)
    targets = torch.cat(query_targets)  # (Q,)

    dists = torch.cdist(q, protos)  # (Q, C)
    logits = -dists
    loss = F.cross_entropy(logits, targets)
    acc = (logits.argmax(dim=1) == targets).float().mean()
    return loss, acc


# ---------------------------------------------------------------------------
# Collapse diagnostics
# ---------------------------------------------------------------------------

def embedding_diagnostics(embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
    """Alignment/uniformity (Wang & Isola 2020) + separation ratio + effective
    rank. The quantitative collapse detector."""
    n = embeddings.shape[0]
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    off = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
    dist = torch.cdist(embeddings, embeddings)

    within = dist[same & off]
    between = dist[~same & off]
    align = (within.pow(2).mean()).item() if within.numel() else 0.0
    # uniformity: log E[exp(-2 * dist^2)] over all off-diagonal pairs
    unif = torch.log(torch.exp(-2.0 * dist[off].pow(2)).mean() + 1e-12).item()
    sep = (between.mean() / within.mean()).item() if within.numel() and within.mean() > 0 else float("nan")

    x = embeddings - embeddings.mean(0, keepdim=True)
    cov = (x.t() @ x) / n
    eig = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    p = eig / eig.sum()
    eff_rank = float(torch.exp(-(p * torch.log(p)).sum()))
    return {"align": align, "uniform": unif, "separation": sep, "eff_rank": eff_rank}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

LICENSE_NOTE = (
    "Trained on human tap data that is substantially the QBT-Extended research "
    "dataset (CC BY-NC-SA 3.0 US, non-commercial only). Derivative work — do "
    "not ship commercially without resolving licensing."
)


def save_encoder_v2(model: RhythmEncoderV2, path, meta: dict | None = None) -> None:
    torch.save(
        {
            "version": 2,
            "state_dict": model.state_dict(),
            "embed_dim": model.embed_dim,
            "backbone": model.backbone_kind,
            "meta": meta or {},
            "license_note": LICENSE_NOTE,
        },
        path,
    )


def load_encoder_v2(path) -> tuple[RhythmEncoderV2, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = RhythmEncoderV2(embed_dim=payload["embed_dim"], backbone=payload.get("backbone", "tcn"))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta", {})
