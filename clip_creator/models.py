"""Data definitions and error types."""

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    words: list[Word]


class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    language: str = "es"
    duration: float


class TopicBoundary(BaseModel):
    timestamp: float
    confidence: float = Field(ge=0.0, le=1.0)


class CandidateSegment(BaseModel):
    start: float
    end: float
    rationale: str


class PipelineOutput(BaseModel):
    episode_file: str
    duration: float
    segments: list[CandidateSegment]
    topic_boundaries: list[TopicBoundary]
    model_used: str
    timestamp: str


class ClipResult(BaseModel):
    path: str
    start: float
    end: float
    duration: float


class RunOutput(BaseModel):
    episode_file: str
    duration: float
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
