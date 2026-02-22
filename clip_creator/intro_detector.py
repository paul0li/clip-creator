"""Detects where the intro/outro music starts and stops in the episode.

The intro plays intro.mp3 from the start but cuts it off at an arbitrary
point, so we can't rely on the reference duration. Instead we compare
the episode to the full reference second-by-second (sliding window of
short chunks) to build a similarity curve, then find where it drops off.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from clip_creator.config import Config
from clip_creator.models import MusicBoundaries, format_timestamp

console = Console(stderr=True)

_SAMPLE_DURATION = 10.0  # seconds of reference to use as matching chunk
_SAMPLE_RATE = 8000  # Hz — low rate saves memory, plenty for music detection
_MIN_GAP_SECONDS = 21  # consecutive seconds below threshold to confirm boundary


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


def _detect_outro(episode, ref, sr: int, ep_sec: float, threshold: float):
    """Detect outro using cross-correlation of a reference chunk against the episode tail."""
    import numpy as np
    from scipy.signal import fftconvolve

    ref_total_sec = len(ref) / sr
    sample_start = max(0.0, (ref_total_sec - _SAMPLE_DURATION) / 2)
    start_frame = int(sample_start * sr)
    sample_frames = int(_SAMPLE_DURATION * sr)
    chunk = ref[start_frame : start_frame + sample_frames]

    if len(chunk) < sr * 3:
        return None

    corr = fftconvolve(episode, chunk[::-1], mode="full")
    valid_start = len(chunk) - 1
    corr = corr[valid_start : valid_start + len(episode)]

    n_seconds = int(len(corr) / sr)
    curve: list[float] = []
    for s in range(n_seconds):
        window = corr[s * sr : (s + 1) * sr]
        curve.append(float(np.max(np.abs(window))))

    if not curve:
        return None

    max_val = max(curve)
    if max_val > 0:
        curve = [v / max_val for v in curve]

    return _find_outro_start(curve, threshold)


def _detect(
    episode_path: str, reference_path: str, threshold: float
) -> MusicBoundaries:
    import librosa
    import numpy as np

    sr = _SAMPLE_RATE

    # Load full reference and episode
    ref, _ = librosa.load(reference_path, sr=sr, mono=True)
    episode, _ = librosa.load(episode_path, sr=sr, mono=True)

    ref_sec = len(ref) / sr
    ep_sec = len(episode) / sr

    if ref_sec < 3:
        console.print("[yellow]Intro reference too short, skipping detection.[/yellow]")
        return MusicBoundaries()

    # Build similarity curve using mel spectrogram cosine similarity.
    # The intro is the same content as intro.mp3 but re-encoded (MP3→video→MP3)
    # so raw waveforms differ. Mel spectrograms compare frequency energy per
    # second and are robust to re-encoding phase shifts. We normalize each
    # frame to unit norm so volume differences don't matter.
    _SMOOTH_WINDOW = 30  # seconds — smooths over silences inside intro.mp3

    n_fft = 2048
    hop = sr  # one mel frame per second
    n_mels = 64

    compare_samples = int(min(ref_sec, ep_sec)) * sr
    ref_mel = np.log1p(librosa.feature.melspectrogram(
        y=ref[:compare_samples], sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    ))
    ep_mel = np.log1p(librosa.feature.melspectrogram(
        y=episode[:compare_samples], sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    ))

    n_frames = min(ref_mel.shape[1], ep_mel.shape[1])
    raw_curve: list[float] = []
    for i in range(n_frames):
        r = ref_mel[:, i]
        e = ep_mel[:, i]
        r_norm = np.linalg.norm(r)
        e_norm = np.linalg.norm(e)
        if r_norm > 0 and e_norm > 0:
            sim = float(np.dot(r, e) / (r_norm * e_norm))
        else:
            # Both silent = match (silence in intro.mp3 = silence in episode)
            sim = 1.0 if r_norm < 1e-6 and e_norm < 1e-6 else 0.0
        raw_curve.append(max(0.0, sim))

    # Smooth with rolling average to bridge silences inside intro.mp3
    curve: list[float] = []
    for i in range(len(raw_curve)):
        start = max(0, i - _SMOOTH_WINDOW // 2)
        end = min(len(raw_curve), i + _SMOOTH_WINDOW // 2 + 1)
        curve.append(sum(raw_curve[start:end]) / (end - start))

    if not curve:
        return MusicBoundaries()

    # Debug output: first N seconds of the comparison region
    console.print(f"[dim]Intro similarity curve (ref={ref_sec:.0f}s, 1 bar = 10s):[/dim]")
    for s in range(0, len(curve), 10):
        val = max(curve[s : s + 10])
        bar = "#" * int(val * 40)
        label = "<<" if val >= threshold else ""
        console.print(f"[dim]  {format_timestamp(s)}  {val:.3f}  {bar} {label}[/dim]")

    # Find where the intro stops matching
    boundaries = MusicBoundaries()
    boundaries.intro_end = _find_intro_end(curve, threshold)

    # Outro detection: use cross-correlation with a chunk from the reference
    # against the tail of the episode (reuse the old approach for outro only)
    boundaries.outro_start = _detect_outro(episode, ref, sr, ep_sec, threshold)

    # Only report outro if it's clearly separate from the intro
    if (
        boundaries.outro_start is not None
        and boundaries.intro_end is not None
        and boundaries.outro_start <= boundaries.intro_end + 60
    ):
        boundaries.outro_start = None

    if boundaries.intro_end:
        console.print(
            f"[green]Intro music ends at ~{format_timestamp(boundaries.intro_end)}[/green]"
        )
    if boundaries.outro_start:
        console.print(
            f"[green]Outro music starts at ~{format_timestamp(boundaries.outro_start)}[/green]"
        )
    if not boundaries.intro_end and not boundaries.outro_start:
        console.print("[dim]No intro/outro music detected above threshold.[/dim]")

    return boundaries
