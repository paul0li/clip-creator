# Clip Creator

Extract the best short clips from podcast episodes automatically.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv` on macOS, `curl -LsSf https://astral.sh/uv/install.sh | sh` on Linux)
- FFmpeg (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Linux)
- An API key for your LLM provider (Anthropic or OpenAI)

## Installation

```bash
# Clone the repo
git clone <repo-url> && cd clip-creator

# Install dependencies (creates .venv, installs Python 3.11 if needed)
uv sync
```

Set your API key as an environment variable (or in a `.env` file):

```bash
export ANTHROPIC_API_KEY=sk-...
# or
export OPENAI_API_KEY=sk-...
```

## Usage

### `run` — Full pipeline (recommended)

Go from video to clips in one command:

```bash
# Produces clips in ./clips and prints JSON output
uv run clip-creator run video.mp4 --output-dir ./clips

# Debug mode — also saves transcript and segments JSON files next to the video
uv run clip-creator run video.mp4 --output-dir ./clips --debug

# Override LLM or Whisper settings
uv run clip-creator run video.mp4 --llm-provider openai --whisper-model large-v3
```

This extracts audio from the video, transcribes it with Whisper, detects jingle boundaries, picks the best moments via LLM, and cuts the clips with FFmpeg.

### Individual steps

You can also run each step separately:

#### `extract` — Extract audio from video

```bash
uv run clip-creator extract video.mp4
```

#### `detect-intro` — Detect intro/outro music

```bash
uv run clip-creator detect-intro episode.mp3
```

#### `transcribe` — Transcribe audio

```bash
uv run clip-creator transcribe episode.mp3

# Use Fireflies instead of local Whisper
uv run clip-creator transcribe episode.mp3 --whisper-mode fireflies --audio-url https://example.com/episode.mp3
```

#### `select` — Pick the best clip segments

```bash
# From a transcript JSON file
uv run clip-creator select transcript.json

# Use OpenAI instead of Anthropic
uv run clip-creator select transcript.json --llm-provider openai
```

The transcript is split into ~10-minute windows, and the LLM picks the best candidate from each window. A final pass selects the top 3 across all windows. This avoids hallucination from sending the full transcript in a single prompt.

#### `cut` — Cut clips from segments

```bash
uv run clip-creator cut video.mp4 --segments segments.json --output-dir ./clips
```

The `--segments` flag accepts either the full JSON output from `select` or a bare list of segments.

## Configuration

Settings are loaded in this order (later overrides earlier):

1. Pydantic defaults
2. `config.yaml`
3. Environment variables (API keys only)
4. CLI flags

See `config.yaml` for available options.
