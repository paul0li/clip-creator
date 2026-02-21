"""Cut video clips using FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from clip_creator.models import CandidateSegment, ClipResult, CutterError


def _check_ffmpeg() -> str:
    """Return path to ffmpeg binary, or raise CutterError."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise CutterError(
            "ffmpeg not found. Install it with: brew install ffmpeg (macOS) "
            "or apt install ffmpeg (Linux)"
        )
    return path


def cut_clips(
    video_path: str | Path,
    segments: list[CandidateSegment],
    output_dir: str | Path = "./clips",
) -> list[ClipResult]:
    """Cut clips from a video file based on segment timestamps.

    Args:
        video_path: Path to the source video file.
        segments: List of CandidateSegment with start/end timestamps.
        output_dir: Directory to write clip files into.

    Returns:
        List of ClipResult, one per segment.

    Raises:
        CutterError: If ffmpeg is missing, input file doesn't exist, or a cut fails.
    """
    ffmpeg = _check_ffmpeg()
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise CutterError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ClipResult] = []

    for i, segment in enumerate(segments, start=1):
        output_file = output_dir / f"clip_{i}.mp4"
        duration = segment.end - segment.start

        cmd = [
            ffmpeg,
            "-y",  # overwrite without asking
            "-ss", str(segment.start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-c:a", "aac",
            str(output_file),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise CutterError(
                f"FFmpeg failed for clip {i} ({segment.start:.1f}s–{segment.end:.1f}s): "
                f"{e.stderr.strip()}"
            ) from e

        if not output_file.exists():
            raise CutterError(f"Expected output file was not created: {output_file}")

        results.append(
            ClipResult(
                path=str(output_file),
                start=segment.start,
                end=segment.end,
                duration=round(duration, 2),
            )
        )

    return results
