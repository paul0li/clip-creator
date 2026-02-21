# Segment Selection — Technical Design

## What this does

Takes a 1-hour podcast episode (audio file) and outputs 3 timestamps for the best moments to cut into short social media clips. That's it.

**Steps:**
1. Transcribe the audio (speech → text with timestamps)
2. Detect where topic changes happen (by finding the jingle sound)
3. Ask an LLM to pick the 3 best 25–40 second moments

---

## Project Structure

```
clip-creator/
├── pyproject.toml              # Project setup and dependencies
├── .env.example                # Template for API keys
├── config.yaml                 # Settings (which model, thresholds, etc.)
├── assets/
│   └── jingle_reference.wav    # The jingle sound to look for
├── src/
│   └── clip_creator/
│       ├── __init__.py
│       ├── cli.py              # Command-line interface
│       ├── models.py           # Data definitions
│       ├── config.py           # Loads settings
│       ├── llm_client.py       # Talks to Claude or OpenAI
│       ├── transcriber.py      # Audio → text (Whisper)
│       ├── jingle_detector.py  # Finds the jingle in the audio
│       ├── segment_selector.py # Asks the LLM to pick best moments
│       └── pipeline.py         # Runs everything in order
└── tests/
    └── ...
```

Every `.py` file does one thing. No sub-packages, no layers of folders.

---

## How to run it

```bash
# Basic usage
clip-creator episode.mp3

# If you already have a transcript (skips the slow transcription step)
clip-creator episode.mp3 --transcript transcript.json

# Save output to a file instead of printing it
clip-creator episode.mp3 --output segments.json
```

Output is JSON printed to the terminal (or a file with `--output`). n8n can pipe it directly.

---

## Output format

```json
{
  "episode_file": "episode.mp3",
  "duration": 3642.5,
  "segments": [
    {
      "start": 187.3,
      "end": 218.9,
      "rationale": "El presentador da una opinion fuerte sobre la reforma tributaria."
    },
    {
      "start": 1045.7,
      "end": 1078.2,
      "rationale": "Reflexion sobre el impacto del cambio climatico en la agricultura local."
    },
    {
      "start": 2834.1,
      "end": 2866.5,
      "rationale": "Momento de humor donde debaten sobre un tema cultural con energia."
    }
  ],
  "topic_boundaries": [
    {"timestamp": 302.5, "confidence": 0.87},
    {"timestamp": 945.2, "confidence": 0.91}
  ],
  "model_used": "anthropic/claude-sonnet-4-20250514",
  "timestamp": "2026-02-21T15:30:00Z"
}
```

---

## Data models (`models.py`)

These define the shape of data passed between modules. Pydantic validates that fields have the right types — especially important for LLM output which can be unpredictable.

```python
class Word(BaseModel):
    text: str
    start: float       # seconds
    end: float

class TranscriptSegment(BaseModel):
    text: str          # a full sentence
    start: float
    end: float
    words: list[Word]

class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    language: str      # "es"
    duration: float    # total seconds

class TopicBoundary(BaseModel):
    timestamp: float   # seconds where the jingle was detected
    confidence: float  # 0.0–1.0

class CandidateSegment(BaseModel):
    start: float
    end: float
    rationale: str     # in Spanish

class PipelineOutput(BaseModel):
    episode_file: str
    duration: float
    segments: list[CandidateSegment]
    topic_boundaries: list[TopicBoundary]
    model_used: str
    timestamp: str
```

`Word` exists only as an intermediate step — Whisper gives us word-level timestamps, we group words into sentences, and after that words are never used again.

---

## Configuration (`config.py`)

Settings come from three places (each overrides the previous):
1. **Defaults** hardcoded in the Config class
2. **`config.yaml`** file
3. **CLI flags** like `--llm-provider openai`

API keys come from environment variables only (never in config files).

```yaml
# config.yaml
whisper:
  mode: local          # "local" or "api"
  model: medium        # tiny/base/small/medium/large-v3
  language: es

jingle:
  reference_path: assets/jingle_reference.wav
  threshold: 0.6

llm:
  provider: anthropic  # "anthropic" or "openai"
  model: claude-sonnet-4-20250514
  temperature: 0.3

segments:
  count: 3
  min_seconds: 25
  max_seconds: 40
```

```
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

## LLM client (`llm_client.py`)

Simple wrapper that works with either Anthropic or OpenAI. Just an if/else — no fancy abstractions.

```python
class LLMClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt, get text back."""
        if self.provider == "anthropic":
            # call Claude API
        elif self.provider == "openai":
            # call OpenAI API
```

---

## Transcription (`transcriber.py`)

**Local mode:** Runs Whisper on your machine. Free but slow (5–15 min for 1 hour).

**API mode:** Sends audio to OpenAI's Whisper API. Fast but costs money. Has a 25 MB upload limit, so a 1-hour file gets split into chunks automatically.

Both modes return word-level timestamps. The `_build_transcript` function then groups words into sentences by splitting on punctuation (`.` `?` `!` `¿` `¡`). This is critical because clips must not start or end mid-sentence.

---

## Jingle detection (`jingle_detector.py`)

Finds where the jingle plays in the episode to mark topic boundaries.

**How it works:**
1. Load the episode and the reference jingle audio
2. Convert both to Mel spectrograms (a visual representation of sound — more robust than comparing raw audio because it handles volume differences)
3. Slide the jingle across the episode and measure similarity at each position
4. Peaks above a threshold (default 0.6) = "jingle found here"
5. Merge detections that are within 5 seconds of each other

**Non-fatal:** If the jingle file is missing or nothing is found, the pipeline keeps going. It just tells the LLM "no topic boundaries detected, figure it out from the content."

---

## Segment selection (`segment_selector.py`)

Sends the transcript + topic boundaries to the LLM and asks it to pick 3 segments.

**The prompt is in Spanish** since the podcast is in Spanish. Key instructions:
- Each segment must be 25–40 seconds
- Must start and end on sentence boundaries
- Must contain something interesting (strong opinion, humor, surprising fact)
- Must be self-contained (makes sense without the rest of the episode)
- All 3 must be about different topics

**Transcript is formatted like this in the prompt:**
```
[00:00:05] Buenos días, bienvenidos al club de las tres de la tarde.
[00:00:09] Hoy vamos a hablar de varios temas importantes.
```

Only sentence timestamps — word-level detail is left out to save tokens.

**If the LLM returns bad JSON:** retry once with "please return only valid JSON." If it still fails, error out.

---

## Pipeline (`pipeline.py`)

Runs the three steps in order:

```python
def run_pipeline(audio_path, config):
    transcript = transcribe(audio_path, config)
    boundaries = detect_jingle_boundaries(audio_path, config)
    segments = select_segments(transcript, boundaries, config)
    return PipelineOutput(...)
```

There's also `run_pipeline_from_transcript` for when you already have a transcript and want to skip the slow transcription step.

---

## Error handling

| What fails | What happens |
|---|---|
| Transcription | Stop. Can't do anything without text. |
| Jingle detection | Keep going with no boundaries. LLM infers topics from content. |
| LLM returns bad JSON | Retry once. If still bad, stop. |
| LLM segments fail validation | Retry once with the error message. If still bad, stop. |

Exit code 0 + JSON on stdout = success. Exit code 1 + error on stderr = failure.

---

## Dependencies

```toml
[project]
name = "clip-creator"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai-whisper>=20231117",  # Local transcription
    "openai>=1.14",              # Whisper API + GPT
    "anthropic>=0.49",           # Claude API
    "librosa>=0.10",             # Audio analysis
    "soundfile>=0.12",           # Audio I/O
    "numpy>=1.26",
    "scipy>=1.12",               # Cross-correlation for jingle matching
    "pydantic>=2.6",             # Data validation
    "pyyaml>=6.0",               # Config file
    "python-dotenv>=1.0",        # .env loading
    "rich>=13.7",                # Progress bars in terminal
]
```

---

## Build order

| Order | File | Why this order |
|---|---|---|
| 1 | `models.py` | Everything else imports these |
| 2 | `config.py` | Small, self-contained |
| 3 | `llm_client.py` | Can test immediately |
| 4 | `segment_selector.py` | Highest risk — prompt needs iteration |
| 5 | `transcriber.py` | Straightforward |
| 6 | `jingle_detector.py` | Pipeline works without it |
| 7 | `pipeline.py` | Wires everything together |
| 8 | `cli.py` | Just argument parsing |
