"""Discrete pipeline steps, each runnable independently or chained via `run`."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from clip_creator.config import Config
from clip_creator.cutter import cut_clips
from clip_creator.intro_detector import detect_music_boundaries
from clip_creator.jingle_detector import detect_jingle_boundaries
from clip_creator.llm_client import LLMClient
from clip_creator.models import (
    CandidateSegment,
    ClipResult,
    MusicBoundaries,
    PipelineOutput,
    RunOutput,
    TopicBoundary,
    Transcript,
)
from clip_creator.segment_selector import select_segments
from clip_creator.transcriber import transcribe


def extract_audio(video_path: str) -> str:
    """Extract audio from video to MP3 via ffmpeg. Returns the MP3 path.

    Saves as {stem}.mp3 next to the video. Skips if the file already exists.
    """
    video = Path(video_path)
    mp3_path = video.with_suffix(".mp3")

    if mp3_path.exists():
        print(f"Audio already exists, skipping extraction: {mp3_path}", file=sys.stderr)
        return str(mp3_path)

    print(f"Extracting audio from {video.name}...", file=sys.stderr)
    subprocess.run(
        [
            "ffmpeg", "-i", str(video),
            "-vn", "-acodec", "libmp3lame", "-q:a", "2",
            str(mp3_path),
        ],
        check=True,
        capture_output=True,
    )
    print(f"Audio saved to {mp3_path}", file=sys.stderr)
    return str(mp3_path)


def step_detect_intro(audio_path: str, config: Config) -> MusicBoundaries:
    """Detect intro/outro music boundaries. Saves {stem}_boundaries.json."""
    boundaries = detect_music_boundaries(audio_path, config)

    out_path = Path(audio_path).with_name(
        Path(audio_path).stem + "_boundaries.json"
    )
    out_path.write_text(boundaries.model_dump_json(indent=2))
    print(f"Boundaries saved to {out_path}", file=sys.stderr)

    return boundaries


def step_transcribe(
    audio_path: str,
    config: Config,
    *,
    boundaries_path: str | None = None,
) -> Transcript:
    """Transcribe audio, trimming intro/outro if boundaries exist.

    Automatically looks for {stem}_boundaries.json next to the audio file
    unless an explicit boundaries_path is given.
    Saves {stem}_transcript.json.
    """
    # Resolve boundaries
    if boundaries_path:
        boundaries = MusicBoundaries.model_validate_json(
            Path(boundaries_path).read_text()
        )
    else:
        auto_path = Path(audio_path).with_name(
            Path(audio_path).stem + "_boundaries.json"
        )
        if auto_path.exists():
            boundaries = MusicBoundaries.model_validate_json(auto_path.read_text())
            print(f"Loaded boundaries from {auto_path}", file=sys.stderr)
        else:
            boundaries = MusicBoundaries()

    transcript = transcribe(
        audio_path,
        config,
        skip_seconds=boundaries.intro_end or 0.0,
        end_seconds=boundaries.outro_start or 0.0,
    )

    out_path = Path(audio_path).with_name(
        Path(audio_path).stem + "_transcript.json"
    )
    out_path.write_text(transcript.model_dump_json(indent=2))
    print(f"Transcript saved to {out_path}", file=sys.stderr)

    return transcript


def step_select(
    transcript_path: str,
    config: Config,
) -> list[CandidateSegment]:
    """Select best segments from a transcript. Saves {stem}_segments.json.

    Looks for jingle boundaries by checking {audio_stem}_boundaries.json
    (inferred from transcript filename: {stem}_transcript.json → {stem}).
    """
    transcript = Transcript.model_validate_json(Path(transcript_path).read_text())

    # Infer the audio stem: remove _transcript suffix
    stem = Path(transcript_path).stem
    if stem.endswith("_transcript"):
        audio_stem = stem[: -len("_transcript")]
    else:
        audio_stem = stem

    # Try to find the audio file for jingle detection
    parent = Path(transcript_path).parent
    audio_path = None
    for ext in (".mp3", ".wav", ".m4a"):
        candidate = parent / (audio_stem + ext)
        if candidate.exists():
            audio_path = str(candidate)
            break

    boundaries: list[TopicBoundary] = []
    if audio_path:
        boundaries = detect_jingle_boundaries(audio_path, config)

    llm_client = LLMClient(config)
    segments = select_segments(transcript, boundaries, config, llm_client)

    # Save segments as PipelineOutput for compatibility with cut --segments
    output = PipelineOutput(
        episode_file=transcript_path,
        duration=transcript.duration,
        segments=segments,
        topic_boundaries=boundaries,
        model_used=llm_client.model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    out_path = parent / f"{audio_stem}_segments.json"
    out_path.write_text(output.model_dump_json(indent=2))
    print(f"Segments saved to {out_path}", file=sys.stderr)

    return segments


def run_full_pipeline(
    video_path: str,
    config: Config,
    output_dir: str = "./clips",
) -> RunOutput:
    """End-to-end: video → audio → boundaries → transcribe → select → cut."""
    audio_path = extract_audio(video_path)
    boundaries = step_detect_intro(audio_path, config)

    transcript = transcribe(
        audio_path,
        config,
        skip_seconds=boundaries.intro_end or 0.0,
        end_seconds=boundaries.outro_start or 0.0,
    )

    # Save transcript
    stem = Path(video_path).stem
    parent = Path(video_path).parent
    transcript_file = parent / f"{stem}_transcript.json"
    transcript_file.write_text(transcript.model_dump_json(indent=2))
    print(f"Transcript saved to {transcript_file}", file=sys.stderr)

    # Jingle detection + segment selection
    jingle_boundaries = detect_jingle_boundaries(audio_path, config)
    llm_client = LLMClient(config)
    segments = select_segments(transcript, jingle_boundaries, config, llm_client)

    # Save segments
    segments_output = PipelineOutput(
        episode_file=video_path,
        duration=transcript.duration,
        segments=segments,
        topic_boundaries=jingle_boundaries,
        model_used=llm_client.model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    segments_file = parent / f"{stem}_segments.json"
    segments_file.write_text(segments_output.model_dump_json(indent=2))
    print(f"Segments saved to {segments_file}", file=sys.stderr)

    # Cut clips
    clips = cut_clips(video_path, segments, output_dir)

    return RunOutput(
        episode_file=video_path,
        duration=transcript.duration,
        segments=segments,
        topic_boundaries=jingle_boundaries,
        model_used=llm_client.model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        clips=clips,
    )
