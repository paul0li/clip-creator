"""Runs transcription, jingle detection, and segment selection in order."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from clip_creator.config import Config, load_config
from clip_creator.cutter import cut_clips
from clip_creator.intro_detector import detect_music_boundaries
from clip_creator.jingle_detector import detect_jingle_boundaries
from clip_creator.llm_client import LLMClient
from clip_creator.models import PipelineOutput, RunOutput, Transcript
from clip_creator.segment_selector import select_segments
from clip_creator.transcriber import load_transcript, transcribe


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


def run_pipeline(
    audio_path: str, config: Config, *, debug: bool = False
) -> PipelineOutput:
    """Full pipeline: transcribe → detect jingles → select segments."""
    music = detect_music_boundaries(audio_path, config)
    transcript = transcribe(
        audio_path, config,
        skip_seconds=music.intro_end or 0.0,
        end_seconds=music.outro_start or 0.0,
    )

    if debug:
        stem = Path(audio_path).stem
        transcript_file = Path(audio_path).parent / f"{stem}_transcript.json"
        transcript_file.write_text(transcript.model_dump_json(indent=2))
        print(f"Transcript saved to {transcript_file}", file=sys.stderr)

    return _run_from_transcript(audio_path, transcript, config)


def run_pipeline_from_transcript(
    audio_path: str, transcript_path: str, config: Config
) -> PipelineOutput:
    """Skip transcription — use a previously saved transcript."""
    transcript = load_transcript(transcript_path)
    return _run_from_transcript(audio_path, transcript, config)


def run_full_pipeline(
    video_path: str,
    config: Config,
    output_dir: str = "./clips",
    *,
    debug: bool = False,
) -> RunOutput:
    """End-to-end: video → audio → transcribe → jingles → segments → cut clips."""
    audio_path = extract_audio(video_path)
    music = detect_music_boundaries(audio_path, config)
    transcript = transcribe(
        audio_path, config,
        skip_seconds=music.intro_end or 0.0,
        end_seconds=music.outro_start or 0.0,
    )

    stem = Path(video_path).stem

    if debug:
        transcript_file = Path(video_path).parent / f"{stem}_transcript.json"
        transcript_file.write_text(transcript.model_dump_json(indent=2))
        print(f"Transcript saved to {transcript_file}", file=sys.stderr)

    boundaries = detect_jingle_boundaries(audio_path, config)
    llm_client = LLMClient(config)
    segments = select_segments(transcript, boundaries, config, llm_client)

    if debug:
        from clip_creator.models import PipelineOutput as _PO

        segments_output = _PO(
            episode_file=video_path,
            duration=transcript.duration,
            segments=segments,
            topic_boundaries=boundaries,
            model_used=llm_client.model_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        segments_file = Path(video_path).parent / f"{stem}_segments.json"
        segments_file.write_text(segments_output.model_dump_json(indent=2))
        print(f"Segments saved to {segments_file}", file=sys.stderr)

    clips = cut_clips(video_path, segments, output_dir)

    return RunOutput(
        episode_file=video_path,
        duration=transcript.duration,
        segments=segments,
        topic_boundaries=boundaries,
        model_used=llm_client.model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        clips=clips,
    )


def _run_from_transcript(
    audio_path: str, transcript: Transcript, config: Config
) -> PipelineOutput:
    boundaries = detect_jingle_boundaries(audio_path, config)
    llm_client = LLMClient(config)
    segments = select_segments(transcript, boundaries, config, llm_client)

    return PipelineOutput(
        episode_file=audio_path,
        duration=transcript.duration,
        segments=segments,
        topic_boundaries=boundaries,
        model_used=llm_client.model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
