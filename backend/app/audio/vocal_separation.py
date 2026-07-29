"""Vocal isolation for auto-tap extraction, via Demucs (Défossez et al.)'s
pretrained htdemucs model.

Why: raw-mix onset detection (app/audio/onsets.py's fallback path) can't
distinguish a kick drum from a sung note — both are just "loud" in the
melody frequency band. That ambiguity is exactly what made the un-isolated
heuristic measure at 0% alignment with real human taps (see
scripts/eval_auto_rhythm.py). Stripping out drums/bass/harmony first removes
most of that ambiguity, since the isolated vocal stem mostly doesn't contain
a kick drum at all.

Cost: real, not hypothetical — Demucs inference is CPU-heavy. On this
machine, ~28s for a 15-second clip. That's why this is only ever invoked
on a short (hook-window) slice, never a whole song, and why callers run it
via FastAPI BackgroundTasks / an offline script rather than inline in a
request a mobile client is waiting on.

Optional dependency: torch + torchaudio + demucs. Every public function can
raise (model download failure, OOM, unsupported audio, ...) — callers MUST
catch and fall back to raw-mix onset detection. This module is never
imported at startup, only lazily when vocal separation is attempted.
"""

import threading

import numpy as np

from app.audio.decode import AudioDecodeError, decode_to_stereo

_model = None
_model_lock = threading.Lock()
_load_failed_reason: str | None = None

DEMUCS_MODEL_NAME = "htdemucs"


class VocalSeparationUnavailable(Exception):
    pass


def _get_model():
    global _model, _load_failed_reason
    if _load_failed_reason is not None:
        raise VocalSeparationUnavailable(_load_failed_reason)
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from demucs.pretrained import get_model

            model = get_model(DEMUCS_MODEL_NAME)
            model.eval()
        except Exception as exc:
            _load_failed_reason = f"could not load {DEMUCS_MODEL_NAME}: {exc}"
            raise VocalSeparationUnavailable(_load_failed_reason) from exc
        _model = model
    return _model


def separate_vocals_window(
    data: bytes, start_seconds: float, end_seconds: float, target_rate: int
) -> np.ndarray:
    """Decode `data`, isolate just [start_seconds, end_seconds), run Demucs
    on that slice only (keeps inference time small), and return the vocal
    stem as float32 mono at `target_rate`.

    Raises VocalSeparationUnavailable if torch/demucs/the model aren't
    usable, or any other exception on decode/inference failure — callers
    must catch and fall back to raw-mix onset detection.
    """
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise VocalSeparationUnavailable(f"torch/torchaudio not installed: {exc}") from exc

    model = _get_model()

    # Decode via PyAV (same decoder as the rest of the app), not
    # torchaudio.load — current torchaudio versions require a separate
    # torchcodec install for file I/O, which we don't otherwise need.
    try:
        stereo = decode_to_stereo(data, model.samplerate)  # (2, samples) at model rate already
    except AudioDecodeError as exc:
        raise VocalSeparationUnavailable(f"could not decode audio for separation: {exc}") from exc

    waveform = torch.from_numpy(stereo)
    start_sample = max(0, int(start_seconds * model.samplerate))
    end_sample = min(waveform.shape[1], int(end_seconds * model.samplerate))
    if end_sample <= start_sample:
        raise VocalSeparationUnavailable("empty time window")
    waveform = waveform[:, start_sample:end_sample]

    from demucs.apply import apply_model

    with torch.no_grad():
        sources = apply_model(model, waveform.unsqueeze(0), progress=False)[0]

    vocals_index = model.sources.index("vocals")
    vocals = sources[vocals_index]  # (channels, samples) at model.samplerate
    vocals_mono = vocals.mean(dim=0, keepdim=True)

    if model.samplerate != target_rate:
        vocals_mono = torchaudio.functional.resample(vocals_mono, model.samplerate, target_rate)

    return vocals_mono.squeeze(0).numpy().astype(np.float32)
