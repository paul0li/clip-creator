"""Data definitions and error types."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.s (with tenths when fractional)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if s == int(s):
        return f"{h:02d}:{m:02d}:{int(s):02d}"
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def _parse_timestamp(value: object) -> float:
    """Accept either a float or an 'HH:MM:SS[.s]' string and return seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", value)
        if m:
            return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
        # Maybe it's a numeric string like "187.3"
        return float(value)
    raise ValueError(f"Cannot parse timestamp: {value!r}")


# A float that serializes to "HH:MM:SS" in JSON but stays a float in Python.
Timestamp = Annotated[
    float,
    BeforeValidator(_parse_timestamp),
    PlainSerializer(format_timestamp, return_type=str),
]


class MusicBoundaries(BaseModel):
    """Timestamps (in seconds) for intro end and outro start."""

    intro_end: Timestamp | None = None
    outro_start: Timestamp | None = None


class Word(BaseModel):
    text: str
    start: Timestamp
    end: Timestamp


class TranscriptSegment(BaseModel):
    text: str
    start: Timestamp
    end: Timestamp
    words: list[Word]


class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    language: str = "es"
    duration: Timestamp


class TopicBoundary(BaseModel):
    timestamp: Timestamp
    confidence: float = Field(ge=0.0, le=1.0)


class CandidateSegment(BaseModel):
    start: Timestamp
    end: Timestamp
    rationale: str


class PipelineOutput(BaseModel):
    episode_file: str
    duration: Timestamp
    segments: list[CandidateSegment]
    topic_boundaries: list[TopicBoundary]
    model_used: str
    timestamp: str


class ClipResult(BaseModel):
    path: str
    start: Timestamp
    end: Timestamp
    duration: Timestamp


class RunOutput(BaseModel):
    episode_file: str
    duration: Timestamp
    segments: list[CandidateSegment]
    topic_boundaries: list[TopicBoundary]
    model_used: str
    timestamp: str
    clips: list[ClipResult]


class TranscriptionError(Exception):
    """Transcription failed — pipeline cannot continue."""


class LLMError(Exception):
    """LLM call or parsing failed after retries."""


class CutterError(Exception):
    """FFmpeg clip cutting failed."""
