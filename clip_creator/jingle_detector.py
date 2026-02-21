"""Finds the jingle sound in the episode using Mel spectrogram cross-correlation."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from clip_creator.config import Config
from clip_creator.models import TopicBoundary

console = Console(stderr=True)


def detect_jingle_boundaries(audio_path: str, config: Config) -> list[TopicBoundary]:
    """Find where the jingle plays in the episode. Non-fatal — returns [] on failure."""
    ref_path = Path(config.jingle.reference_path)
    if not ref_path.is_absolute():
        ref_path = Path(__file__).resolve().parent.parent / ref_path

    if not ref_path.exists():
        console.print(
            f"[yellow]Jingle reference not found at {ref_path}, skipping detection.[/yellow]"
        )
        return []

    try:
        return _detect(audio_path, str(ref_path), config.jingle.threshold)
    except Exception as e:
        console.print(f"[yellow]Jingle detection failed: {e}. Continuing without it.[/yellow]")
        return []


def _detect(
    episode_path: str, reference_path: str, threshold: float
) -> list[TopicBoundary]:
    import librosa
    import numpy as np
    from scipy.signal import fftconvolve

    # Load audio as mono
    episode, sr = librosa.load(episode_path, sr=22050, mono=True)
    reference, _ = librosa.load(reference_path, sr=22050, mono=True)

    # Convert to Mel spectrograms
    ep_mel = librosa.feature.melspectrogram(y=episode, sr=sr, n_mels=128)
    ref_mel = librosa.feature.melspectrogram(y=reference, sr=sr, n_mels=128)

    # Log scale
    ep_mel = librosa.power_to_db(ep_mel, ref=np.max)
    ref_mel = librosa.power_to_db(ref_mel, ref=np.max)

    # Cross-correlation: average across mel bands
    # Flatten mel bands by averaging, then correlate
    ep_flat = ep_mel.mean(axis=0)
    ref_flat = ref_mel.mean(axis=0)

    # Normalize
    ep_flat = (ep_flat - ep_flat.mean()) / (ep_flat.std() + 1e-8)
    ref_flat = (ref_flat - ref_flat.mean()) / (ref_flat.std() + 1e-8)

    # Cross-correlation via FFT
    correlation = fftconvolve(ep_flat, ref_flat[::-1], mode="valid")
    correlation = correlation / len(ref_flat)  # Normalize to ~[-1, 1]

    # Find peaks above threshold
    hop_length = 512  # librosa default
    frames_to_seconds = hop_length / sr

    boundaries: list[TopicBoundary] = []
    for i, val in enumerate(correlation):
        if val >= threshold:
            timestamp = i * frames_to_seconds
            boundaries.append(TopicBoundary(timestamp=timestamp, confidence=min(float(val), 1.0)))

    # Merge detections within 5 seconds of each other
    boundaries = _merge_nearby(boundaries, min_gap=5.0)

    console.print(f"[green]Found {len(boundaries)} jingle occurrence(s).[/green]")
    return boundaries


def _merge_nearby(
    boundaries: list[TopicBoundary], min_gap: float
) -> list[TopicBoundary]:
    """Keep only the highest-confidence detection within each cluster."""
    if not boundaries:
        return []

    boundaries.sort(key=lambda b: b.timestamp)
    merged: list[TopicBoundary] = [boundaries[0]]

    for b in boundaries[1:]:
        if b.timestamp - merged[-1].timestamp < min_gap:
            # Keep the one with higher confidence
            if b.confidence > merged[-1].confidence:
                merged[-1] = b
        else:
            merged.append(b)

    return merged
