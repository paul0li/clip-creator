"""Detects where the intro/outro music plays in the episode audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from clip_creator.config import Config

console = Console(stderr=True)


@dataclass
class MusicBoundaries:
    """Timestamps (in seconds) for intro end and outro start."""

    intro_end: float | None = None
    outro_start: float | None = None


def detect_music_boundaries(audio_path: str, config: Config) -> MusicBoundaries:
    """Find where the intro/outro music plays. Returns boundaries or empty defaults."""
    ref_path = Path(config.intro.reference_path)
    if not ref_path.is_absolute():
        ref_path = Path(__file__).resolve().parent.parent / ref_path

    if not ref_path.exists():
        console.print(
            f"[yellow]Intro reference not found at {ref_path}, skipping detection.[/yellow]"
        )
        return MusicBoundaries()

    try:
        return _detect(audio_path, str(ref_path), config.intro.threshold)
    except Exception as e:
        console.print(f"[yellow]Music detection failed: {e}. Continuing without it.[/yellow]")
        return MusicBoundaries()


def _detect(
    episode_path: str, reference_path: str, threshold: float
) -> MusicBoundaries:
    import librosa
    import numpy as np
    from scipy.signal import fftconvolve

    episode, sr = librosa.load(episode_path, sr=22050, mono=True)
    reference, _ = librosa.load(reference_path, sr=22050, mono=True)

    ref_duration = len(reference) / sr

    # Mel spectrograms
    ep_mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=episode, sr=sr, n_mels=128), ref=np.max
    )
    ref_mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=reference, sr=sr, n_mels=128), ref=np.max
    )

    # Average across mel bands and normalize
    ep_flat = ep_mel.mean(axis=0)
    ref_flat = ref_mel.mean(axis=0)
    ep_flat = (ep_flat - ep_flat.mean()) / (ep_flat.std() + 1e-8)
    ref_flat = (ref_flat - ref_flat.mean()) / (ref_flat.std() + 1e-8)

    # Cross-correlation
    correlation = fftconvolve(ep_flat, ref_flat[::-1], mode="valid")
    correlation = correlation / len(ref_flat)

    hop_length = 512
    frames_to_seconds = hop_length / sr

    # Find all peaks above threshold
    matches: list[tuple[float, float]] = []  # (start_time, confidence)
    above = correlation >= threshold
    i = 0
    while i < len(above):
        if above[i]:
            # Find the best value in this cluster
            best_val = correlation[i]
            best_idx = i
            while i < len(above) and above[i]:
                if correlation[i] > best_val:
                    best_val = correlation[i]
                    best_idx = i
                i += 1
            matches.append((best_idx * frames_to_seconds, float(best_val)))
        else:
            i += 1

    if not matches:
        console.print("[dim]No intro/outro music detected above threshold.[/dim]")
        return MusicBoundaries()

    boundaries = MusicBoundaries()

    # First match → intro
    intro_start, intro_conf = matches[0]
    boundaries.intro_end = intro_start + ref_duration
    console.print(
        f"[green]Intro detected at {intro_start:.1f}s–{boundaries.intro_end:.1f}s "
        f"(confidence: {intro_conf:.2f})[/green]"
    )

    # Last match → outro (only if it's a different match from the intro)
    if len(matches) >= 2:
        outro_start, outro_conf = matches[-1]
        boundaries.outro_start = outro_start
        outro_end = outro_start + ref_duration
        console.print(
            f"[green]Outro detected at {outro_start:.1f}s–{outro_end:.1f}s "
            f"(confidence: {outro_conf:.2f})[/green]"
        )

    return boundaries
