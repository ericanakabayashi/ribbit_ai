"""BirdNET embedding extraction.

The model is loaded lazily on first call so import cost is zero at startup.
BirdNET-Analyzer must be installed or on sys.path before calling these functions.
"""

import io
import numpy as np
import librosa

_birdnet_embeddings = None


def _load_birdnet():
    global _birdnet_embeddings
    if _birdnet_embeddings is None:
        from birdnet_analyzer.model import load_model, embeddings as _emb

        load_model(class_output=False)
        _birdnet_embeddings = _emb
    return _birdnet_embeddings


def extract_embeddings(audio_data: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """Return a (1024,) BirdNET embedding from a numpy audio array."""
    emb_fn = _load_birdnet()
    audio = np.array(audio_data, dtype=np.float32)
    target = sample_rate * 3
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)), "constant")
    else:
        audio = audio[:target]
    return emb_fn(np.expand_dims(audio, axis=0)).flatten()


def extract_embeddings_from_bytes(audio_bytes: bytes, sample_rate: int = 48000) -> np.ndarray:
    """Return a (1024,) BirdNET embedding from raw audio bytes (WAV/MP3/etc.)."""
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    return extract_embeddings(audio, sample_rate)
