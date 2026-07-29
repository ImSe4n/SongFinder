"""Learned rhythm-matching: a small metric-learning encoder that maps a tap
sequence to a fixed-size embedding, trained so that different people tapping
the SAME song land close together and different songs land far apart.

This is the first actual machine-learning component in SongFinder — unlike
the DTW matcher (app/rhythm/matching.py) and the onset heuristic
(app/audio/onsets.py), which are fixed hand-written algorithms, this one's
notion of "similar rhythm" is learned from real human tap data.

Input representation: the SAME tempo-invariant log-ratio interval sequence
the DTW matcher uses (app/rhythm/features.py) — log(ioi[i+1]/ioi[i]) —
so tempo invariance is baked in by construction rather than something the
model has to learn. Sequences are variable-length; the encoder is a small
1-D CNN with masked mean+max pooling, so any length >= 1 works.

Training objective: batch-hard triplet loss on L2-normalized embeddings
(Hermans et al. 2017 "In Defense of the Triplet Loss") — within each batch,
for every sample, push away the nearest wrong-song sample and pull in the
farthest same-song sample.

Training data provenance / licensing: the bulk of the human tap corpus is
imported from the QBT-Extended research dataset (CC BY-NC-SA, non-commercial
only — see scripts/import_qbt_extended.py). A model trained on it is a
derivative work: keep checkpoints out of git (backend/models/ is ignored)
and re-check licensing before any commercial use.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.rhythm.features import iois_to_log_ratio, taps_to_iois

# Log-ratios beyond this magnitude mean a >12x tempo change between adjacent
# gaps — that's a recording glitch or a fat-finger, not rhythm. Clipping keeps
# single outliers from dominating the convolution activations.
LOG_RATIO_CLIP = 2.5

EMBED_DIM = 64


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

def random_crop_taps(taps_ms: list[int], rng: random.Random, min_taps: int = 8) -> list[int]:
    """Random contiguous crop of a tap sequence. Two jobs: (a) simulate the
    short queries real app users produce (~6-15 taps vs QBT-Extended's ~33),
    and (b) simulate starting partway into the hook — something DTW's global
    alignment is genuinely bad at, and a potential edge for the model."""
    n = len(taps_ms)
    if n <= min_taps:
        return taps_ms
    length = rng.randint(min_taps, n)
    start = rng.randint(0, n - length)
    return taps_ms[start : start + length]


def head_crop_taps(taps_ms: list[int], n_taps: int = 12) -> list[int]:
    """Deterministic 'short query' for evaluation: the first n taps —
    roughly how a real user taps the start of a hook and stops."""
    return taps_ms[:n_taps]


def featurize(
    taps_ms: list[int],
    jitter: float = 0.0,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """taps (ms, relative) -> clipped log-ratio sequence as a float tensor.

    `jitter` multiplies each inter-onset interval by (1 ± jitter) noise —
    simulating human timing wobble — before the (tempo-invariant) log-ratio
    transform. Zero jitter reproduces exactly what the DTW matcher sees.
    """
    iois = taps_to_iois(taps_ms)
    if jitter > 0.0 and rng is not None:
        iois = [max(1.0, ioi * (1.0 + rng.uniform(-jitter, jitter))) for ioi in iois]
    ratios = iois_to_log_ratio(iois)
    arr = np.clip(np.asarray(ratios, dtype=np.float32), -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    return torch.from_numpy(arr)


def collate_sequences(
    features: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length 1-D feature tensors into a (B, L) batch
    plus a (B, L) boolean validity mask (True = real data, False = padding)."""
    max_len = max(int(f.shape[0]) for f in features)
    batch = torch.zeros(len(features), max_len, dtype=torch.float32)
    mask = torch.zeros(len(features), max_len, dtype=torch.bool)
    for i, f in enumerate(features):
        n = int(f.shape[0])
        batch[i, :n] = f
        mask[i, :n] = True
    return batch, mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RhythmEncoder(nn.Module):
    """~27k parameters — deliberately tiny. The corpus is ~3k samples; a
    bigger model would just memorize it, and this has to run fast on CPU."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.convs = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L) log-ratio values, mask: (B, L) bool. Returns (B, D)
        L2-normalized embeddings."""
        h = self.convs(x.unsqueeze(1))  # (B, C, L)
        m = mask.unsqueeze(1)  # (B, 1, L)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pool = h.masked_fill(~m, 0.0).sum(dim=2) / valid_counts
        max_pool = h.masked_fill(~m, float("-inf")).amax(dim=2)
        z = self.proj(torch.cat([mean_pool, max_pool], dim=1))
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def embed(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Convenience inference path: featurized sequences -> embeddings."""
        self.eval()
        batch, mask = collate_sequences(features)
        return self.forward(batch, mask)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3
) -> torch.Tensor:
    """For each anchor: hardest (farthest) positive and hardest (nearest)
    negative within the batch. Requires >=2 samples of some class and >=2
    classes per batch (the PK sampler guarantees this)."""
    dist = torch.cdist(embeddings, embeddings)  # (B, B)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)

    hardest_positive = (dist * same.float()).max(dim=1).values  # self-dist 0 never wins
    hardest_negative = dist.masked_fill(same, float("inf")).min(dim=1).values

    return F.relu(hardest_positive - hardest_negative + margin).mean()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@dataclass
class CheckpointInfo:
    embed_dim: int
    eval_results: dict
    license_note: str


LICENSE_NOTE = (
    "Trained on human tap data that is substantially the QBT-Extended research "
    "dataset (CC BY-NC-SA 3.0 US, non-commercial only). This checkpoint is a "
    "derivative work — do not ship commercially without resolving licensing."
)


def save_encoder(model: RhythmEncoder, path, eval_results: dict | None = None) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embed_dim": model.embed_dim,
            "log_ratio_clip": LOG_RATIO_CLIP,
            "eval_results": eval_results or {},
            "license_note": LICENSE_NOTE,
        },
        path,
    )


def load_encoder(path) -> tuple[RhythmEncoder, CheckpointInfo]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = RhythmEncoder(embed_dim=payload["embed_dim"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    info = CheckpointInfo(
        embed_dim=payload["embed_dim"],
        eval_results=payload.get("eval_results", {}),
        license_note=payload.get("license_note", ""),
    )
    return model, info
