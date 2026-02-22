"""Detects where the intro/outro music starts and stops in the episode.

The reference file can be any length (even the full loop). We take a short
sample from it and slide it across the episode to build a similarity curve.
Where the curve drops below threshold = music stopped (intro end).
Where it rises above threshold near the end = music started again (outro start).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from clip_creator.config import Config

console = Console(stderr=True)

# How many seconds of the reference to use as the matching sample
_SAMPLE_DURATION = 30.0

# Step size (in seconds) for sliding the sample across the episode
_STEP_SECONDS = 5.0


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

    sr = 22050

    # Load a short sample from the middle of the reference (avoids fade-in/out)
    ref_full, _ = librosa.load(reference_path, sr=sr, mono=True)
    ref_total_sec = len(ref_full) / sr
    sample_start = min(30.0, ref_total_sec / 3)  # skip first 30s or 1/3
    sample_frames = int(_SAMPLE_DURATION * sr)
    start_frame = int(sample_start * sr)
    reference = ref_full[start_frame : start_frame + sample_frames]

    if len(reference) < sr * 5:
        console.print("[yellow]Intro reference too short, skipping detection.[/yellow]")
        return MusicBoundaries()

    # Build mel spectrogram for the reference sample
    ref_mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=reference, sr=sr, n_mels=128), ref=np.max
    )
    ref_flat = ref_mel.mean(axis=0)
    ref_flat = (ref_flat - ref_flat.mean()) / (ref_flat.std() + 1e-8)

    # Load the episode
    episode, _ = librosa.load(episode_path, sr=sr, mono=True)
    ep_total_sec = len(episode) / sr

    # Slide the sample across the episode and compute similarity at each position
    hop_length = 512
    step_frames = int(_STEP_SECONDS * sr)
    sample_len = len(reference)

    scores: list[tuple[float, float]] = []  # (timestamp, similarity)

    for offset in range(0, len(episode) - sample_len + 1, step_frames):
        chunk = episode[offset : offset + sample_len]
        chunk_mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128), ref=np.max
        )
        chunk_flat = chunk_mel.mean(axis=0)
        chunk_flat = (chunk_flat - chunk_flat.mean()) / (chunk_flat.std() + 1e-8)

        # Use cross-correlation peak as similarity score
        corr = fftconvolve(chunk_flat, ref_flat[::-1], mode="full")
        similarity = float(np.max(corr) / len(ref_flat))

        timestamp = offset / sr
        scores.append((timestamp, similarity))

    if not scores:
        return MusicBoundaries()

    # Log the similarity curve for debugging (first 20 min worth of positions)
    max_debug = int(20 * 60 / _STEP_SECONDS)
    console.print("[dim]Music similarity scores (start of episode):[/dim]")
    for ts, sim in scores[:max_debug]:
        bar = "#" * int(sim * 40)
        label = "<<" if sim >= threshold else ""
        console.print(f"[dim]  {ts:7.1f}s  {sim:.3f}  {bar} {label}[/dim]")
    if len(scores) > max_debug:
        console.print(f"[dim]  ... ({len(scores)} total positions)[/dim]")

    boundaries = MusicBoundaries()

    # Find intro end: scan from the start, find the last position above threshold
    # before a sustained drop below threshold
    for i, (ts, sim) in enumerate(scores):
        if sim >= threshold:
            # This position has music — the intro ends after this chunk
            boundaries.intro_end = ts + _SAMPLE_DURATION
        else:
            # First drop below threshold after finding music = intro ended
            if boundaries.intro_end is not None:
                break

    if boundaries.intro_end:
        console.print(
            f"[green]Intro music ends at ~{boundaries.intro_end:.1f}s[/green]"
        )

    # Find outro start: scan from the end, find the last position above threshold
    for i in range(len(scores) - 1, -1, -1):
        ts, sim = scores[i]
        if sim >= threshold:
            boundaries.outro_start = ts
        else:
            if boundaries.outro_start is not None:
                break

    # Only report outro if it's clearly separate from the intro
    if (
        boundaries.outro_start is not None
        and boundaries.intro_end is not None
        and boundaries.outro_start <= boundaries.intro_end + 60
    ):
        # Outro overlaps with intro — there's no separate outro
        boundaries.outro_start = None

    if boundaries.outro_start:
        console.print(
            f"[green]Outro music starts at ~{boundaries.outro_start:.1f}s[/green]"
        )

    if not boundaries.intro_end and not boundaries.outro_start:
        console.print("[dim]No intro/outro music detected above threshold.[/dim]")

    return boundaries
