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
uv run clip-creator run clip-creator/assets/video.mp4 --output-dir ./clips

# Debug mode — also saves transcript and segments JSON files next to the video
uv run clip-creator run video.mp4 --output-dir ./clips --debug

# Override LLM or Whisper settings
uv run clip-creator run video.mp4 --llm-provider openai --whisper-model large-v3
```

This extracts audio from the video, transcribes it with Whisper, detects jingle boundaries, picks the 3 best moments via LLM, and cuts the clips with FFmpeg.

### `select` — Pick segments only

Transcribe an episode and pick the 3 best clip-worthy moments (no cutting):

```bash
# Basic — transcribes with Whisper, then selects segments
uv run clip-creator select episode.mp3

# Use OpenAI instead of Anthropic
uv run clip-creator select episode.mp3 --llm-provider openai

# Save output to a file
uv run clip-creator select episode.mp3 -o segments.json

# Skip transcription if you already have a transcript
uv run clip-creator select episode.mp3 --transcript transcript.json

# Debug mode — saves transcript JSON next to the audio file
uv run clip-creator select episode.mp3 --debug
```

### `cut` — Cut clips from segments

Take the segments from `select` and cut actual video clips with FFmpeg:

```bash
# Cut clips from a video using the segment output
uv run clip-creator cut video.mp4 --segments segments.json --output-dir ./clips
```

This produces `clip_1.mp4`, `clip_2.mp4`, `clip_3.mp4` in the output directory.

The `--segments` flag accepts either the full JSON output from `select` or a bare list of segments.

### Backward compatibility

You can also run without a subcommand — it defaults to `select`:

```bash
uv run clip-creator episode.mp3
```
