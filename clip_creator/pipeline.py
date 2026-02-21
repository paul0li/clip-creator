"""Runs transcription, jingle detection, and segment selection in order."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from clip_creator.config import Config
from clip_creator.jingle_detector import detect_jingle_boundaries
from clip_creator.llm_client import LLMClient
from clip_creator.models import PipelineOutput, Transcript
from clip_creator.segment_selector import select_segments
from clip_creator.transcriber import load_transcript, transcribe


def run_pipeline(audio_path: str, config: Config) -> PipelineOutput:
    """Full pipeline: transcribe → detect jingles → select segments."""
    transcript = transcribe(audio_path, config)

    # Save transcript for inspection / reuse
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
