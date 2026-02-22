"""Audio → text via Whisper (local or API)."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from rich.console import Console

from clip_creator.config import Config
from clip_creator.models import Transcript, TranscriptSegment, TranscriptionError, Word

console = Console(stderr=True)

# Spanish sentence-ending punctuation
_SENTENCE_END = re.compile(r"[.?!¿¡]$")


def transcribe(
    audio_path: str,
    config: Config,
    *,
    skip_seconds: float = 0.0,
    end_seconds: float = 0.0,
) -> Transcript:
    """Transcribe an audio file, returning sentence-level segments.

    If skip_seconds > 0, trims that many seconds from the start (intro).
    If end_seconds > 0, stops at that timestamp (outro).
    Timestamps are offset back so they match the original audio.
    """
    needs_trim = skip_seconds > 0 or end_seconds > 0
    source_path = audio_path

    if needs_trim:
        source_path = _trim_audio(audio_path, skip_seconds, end_seconds)
        parts = []
        if skip_seconds > 0:
            parts.append(f"intro ({skip_seconds:.1f}s)")
        if end_seconds > 0:
            parts.append(f"outro (from {end_seconds:.1f}s)")
        console.print(f"[dim]Trimmed {' and '.join(parts)} before transcribing.[/dim]")

    if config.whisper.mode == "local":
        words, duration = _transcribe_local(source_path, config)
    elif config.whisper.mode == "api":
        words, duration = _transcribe_api(source_path, config)
    else:
        raise TranscriptionError(f"Unknown whisper mode: {config.whisper.mode}")

    if not words:
        raise TranscriptionError("Whisper returned no words")

    # Offset timestamps back to match the original audio
    if skip_seconds > 0:
        for w in words:
            w.start += skip_seconds
            w.end += skip_seconds
        duration += skip_seconds

    if needs_trim:
        Path(source_path).unlink(missing_ok=True)

    segments = _group_words_into_sentences(words)
    return Transcript(segments=segments, language=config.whisper.language, duration=duration)


def load_transcript(path: str) -> Transcript:
    """Load a previously saved transcript from JSON."""
    return Transcript.model_validate_json(Path(path).read_text())


def _trim_audio(audio_path: str, start: float, end: float) -> str:
    """Trim audio: skip before `start`, stop at `end`. Returns path to trimmed file."""
    import subprocess

    trimmed = Path(audio_path).with_stem(Path(audio_path).stem + "_trimmed")
    cmd = ["ffmpeg", "-i", audio_path]
    if start > 0:
        cmd += ["-ss", str(start)]
    if end > 0:
        cmd += ["-to", str(end - start if start > 0 else end)]
    cmd += ["-acodec", "copy", "-y", str(trimmed)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise TranscriptionError(f"ffmpeg trim failed: {result.stderr}")
    return str(trimmed)


def _transcribe_local(audio_path: str, config: Config) -> tuple[list[Word], float]:
    """Run Whisper locally."""
    try:
        import whisper
    except ImportError as e:
        raise TranscriptionError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        ) from e

    console.print(f"[bold]Loading Whisper model '{config.whisper.model}'...[/bold]")
    model = whisper.load_model(config.whisper.model)

    console.print("[bold]Transcribing (this may take a while)...[/bold]")
    result = model.transcribe(
        audio_path,
        language=config.whisper.language,
        word_timestamps=True,
    )

    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append(Word(text=w["word"].strip(), start=w["start"], end=w["end"]))

    duration = result["segments"][-1]["end"] if result["segments"] else 0.0
    return words, duration


def _transcribe_api(audio_path: str, config: Config) -> tuple[list[Word], float]:
    """Use OpenAI's Whisper API. Converts to MP3 first if file is over 25 MB."""
    try:
        import openai
    except ImportError as e:
        raise TranscriptionError(
            "openai is not installed. Run: pip install openai"
        ) from e

    client = openai.OpenAI(api_key=config.openai_api_key)
    file_size = Path(audio_path).stat().st_size

    # Under 25 MB — send directly
    if file_size <= 25 * 1024 * 1024:
        return _transcribe_api_send(client, audio_path)

    # Over 25 MB — convert to compressed MP3 first
    console.print("[bold]File is large, converting to MP3...[/bold]")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        _convert_to_mp3(audio_path, tmp.name)
        mp3_size = Path(tmp.name).stat().st_size
        console.print(f"[dim]Compressed: {file_size // (1024*1024)} MB → {mp3_size // (1024*1024)} MB[/dim]")
        return _transcribe_api_send(client, tmp.name)


def _convert_to_mp3(input_path: str, output_path: str) -> None:
    """Convert any audio/video file to MP3 using ffmpeg."""
    import subprocess

    result = subprocess.run(
        ["ffmpeg", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000",
         "-ab", "32k", "-y", output_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise TranscriptionError(
            f"ffmpeg conversion failed: {result.stderr}\n"
            "Make sure ffmpeg is installed: brew install ffmpeg"
        )


def _transcribe_api_send(
    client, audio_path: str
) -> tuple[list[Word], float]:
    console.print("[bold]Sending to Whisper API...[/bold]")
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    words = [
        Word(text=w.word.strip(), start=w.start, end=w.end)
        for w in (response.words or [])
    ]
    duration = response.duration or 0.0
    return words, duration


def _group_words_into_sentences(words: list[Word]) -> list[TranscriptSegment]:
    """Group words into sentences by splitting on Spanish punctuation."""
    segments: list[TranscriptSegment] = []
    current_words: list[Word] = []

    for word in words:
        current_words.append(word)
        if _SENTENCE_END.search(word.text):
            segments.append(
                TranscriptSegment(
                    text=" ".join(w.text for w in current_words),
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    words=current_words,
                )
            )
            current_words = []

    # Leftover words that didn't end with punctuation
    if current_words:
        segments.append(
            TranscriptSegment(
                text=" ".join(w.text for w in current_words),
                start=current_words[0].start,
                end=current_words[-1].end,
                words=current_words,
            )
        )

    return segments
