import io

import av
import numpy as np

# 11025 Hz mono is plenty for fingerprinting (music energy of interest is
# well below 5.5 kHz) and keeps spectrogram + hash sizes small.
TARGET_SAMPLE_RATE = 11_025


class AudioDecodeError(Exception):
    pass


def decode_to_mono(data: bytes, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode any container/codec FFmpeg understands (m4a/AAC from Expo,
    wav, mp3, ...) to a float32 mono waveform at `target_rate`."""
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioDecodeError("No audio stream found in upload")

            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_rate)
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            # flush the resampler
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))
    except AudioDecodeError:
        raise
    except Exception as exc:  # av raises a zoo of error types
        raise AudioDecodeError(f"Could not decode audio: {exc}") from exc

    if not chunks:
        raise AudioDecodeError("Audio stream decoded to zero samples")

    return np.concatenate(chunks).astype(np.float32)


def decode_to_stereo(data: bytes, target_rate: int) -> np.ndarray:
    """Decode to float32 stereo, shape (2, samples), at `target_rate`. For
    feeding models that expect real stereo input at a specific rate (e.g.
    Demucs) rather than the downsampled mono used for fingerprinting/onset
    detection. Deliberately reuses PyAV (the same decoder as decode_to_mono)
    rather than torchaudio.load, which — in current torchaudio versions —
    needs an additional torchcodec dependency we don't otherwise need."""
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioDecodeError("No audio stream found in upload")

            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_rate)
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray())
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray())
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(f"Could not decode audio: {exc}") from exc

    if not chunks:
        raise AudioDecodeError("Audio stream decoded to zero samples")

    return np.concatenate(chunks, axis=1).astype(np.float32)
