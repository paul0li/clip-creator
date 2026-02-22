"""Detects where the intro/outro music starts and stops in the episode.

Uses raw waveform cross-correlation — since the intro music is the same source
file every time (no re-encoding), this is both simpler and more accurate than
spectrogram-based approaches.

We take a short chunk from the middle of the reference and correlate it against
the entire episode in one FFT pass, then build a 1-second similarity curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from clip_creator.config import Config

console = Console(stderr=True)

_SAMPLE_DURATION = 10.0  # seconds of reference to use as matching chunk
_SAMPLE_RATE = 8000  # Hz — low rate saves memory, plenty for music detection
_MIN_GAP_SECONDS = 10  # consecutive seconds below threshold to confirm boundary


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


def _find_intro_end(curve: list[float], threshold: float) -> float | None:
    """Scan forward to find the last second above threshold before a sustained gap."""
    last_above: int | None = None
    gap = 0

    for i, val in enumerate(curve):
        if val >= threshold:
            last_above = i
            gap = 0
        else:
            if last_above is not None:
                gap += 1
                if gap >= _MIN_GAP_SECONDS:
                    return float(last_above + 1)  # music ends after that second
    return None


def _find_outro_start(curve: list[float], threshold: float) -> float | None:
    """Scan backward from the end to find where the outro music begins."""
    last_above: int | None = None
    gap = 0

    for i in range(len(curve) - 1, -1, -1):
        if curve[i] >= threshold:
            last_above = i
            gap = 0
        else:
            if last_above is not None:
                gap += 1
                if gap >= _MIN_GAP_SECONDS:
                    return float(last_above)
    return None


def _detect(
    episode_path: str, reference_path: str, threshold: float
) -> MusicBoundaries:
    import librosa
    import numpy as np
    from scipy.signal import fftconvolve

    sr = _SAMPLE_RATE

    # Load a short chunk from the middle of the reference (avoids fade-in/out)
    ref_full, _ = librosa.load(reference_path, sr=sr, mono=True)
    ref_total_sec = len(ref_full) / sr
    sample_start = max(0.0, (ref_total_sec - _SAMPLE_DURATION) / 2)
    start_frame = int(sample_start * sr)
    sample_frames = int(_SAMPLE_DURATION * sr)
    chunk = ref_full[start_frame : start_frame + sample_frames]

    if len(chunk) < sr * 3:
        console.print("[yellow]Intro reference too short, skipping detection.[/yellow]")
        return MusicBoundaries()

    # Load the episode
    episode, _ = librosa.load(episode_path, sr=sr, mono=True)
    ep_total_sec = len(episode) / sr

    # Single FFT cross-correlation of the chunk against the entire episode
    corr = fftconvolve(episode, chunk[::-1], mode="full")

    # The valid region starts at index (len(chunk) - 1) — that's where the chunk
    # is fully overlapping with the episode starting at sample 0
    valid_start = len(chunk) - 1
    corr = corr[valid_start : valid_start + len(episode)]

    # Build 1-second resolution similarity curve: max correlation per second
    n_seconds = int(len(corr) / sr)
    curve: list[float] = []
    for s in range(n_seconds):
        window = corr[s * sr : (s + 1) * sr]
        curve.append(float(np.max(np.abs(window))))

    if not curve:
        return MusicBoundaries()

    # Normalize to [0, 1]
    max_val = max(curve)
    if max_val > 0:
        curve = [v / max_val for v in curve]

    # Debug output: every 10s for first 20 min + last 5 min
    console.print("[dim]Music similarity curve (1 bar = 10s):[/dim]")
    debug_start = min(20 * 60, len(curve))
    for s in range(0, debug_start, 10):
        val = max(curve[s : s + 10])
        bar = "#" * int(val * 40)
        label = "<<" if val >= threshold else ""
        console.print(f"[dim]  {s:5d}s  {val:.3f}  {bar} {label}[/dim]")

    if len(curve) > 20 * 60:
        console.print("[dim]  ...[/dim]")

    tail_start = max(debug_start, len(curve) - 5 * 60)
    for s in range(tail_start, len(curve), 10):
        val = max(curve[s : min(s + 10, len(curve))])
        bar = "#" * int(val * 40)
        label = "<<" if val >= threshold else ""
        console.print(f"[dim]  {s:5d}s  {val:.3f}  {bar} {label}[/dim]")

    # Find boundaries
    boundaries = MusicBoundaries()
    boundaries.intro_end = _find_intro_end(curve, threshold)
    boundaries.outro_start = _find_outro_start(curve, threshold)

    # Only report outro if it's clearly separate from the intro
    if (
        boundaries.outro_start is not None
        and boundaries.intro_end is not None
        and boundaries.outro_start <= boundaries.intro_end + 60
    ):
        boundaries.outro_start = None

    if boundaries.intro_end:
        console.print(
            f"[green]Intro music ends at ~{boundaries.intro_end:.1f}s[/green]"
        )
    if boundaries.outro_start:
        console.print(
            f"[green]Outro music starts at ~{boundaries.outro_start:.1f}s[/green]"
        )
    if not boundaries.intro_end and not boundaries.outro_start:
        console.print("[dim]No intro/outro music detected above threshold.[/dim]")

    return boundaries
