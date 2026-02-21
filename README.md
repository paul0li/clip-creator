# Clip Creator

Extract the best short clips from podcast episodes automatically.

## Prerequisites

- Python 3.11+
- FFmpeg (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Linux)
- An API key for your LLM provider (Anthropic or OpenAI)

## Installation

```bash
# Clone the repo
git clone <repo-url> && cd clip-creator

# Install in editable mode
pip install -e .
```

Set your API key as an environment variable (or in a `.env` file):

```bash
export ANTHROPIC_API_KEY=sk-...
# or
export OPENAI_API_KEY=sk-...
```

## Usage

### Step 1: Select segments

Transcribe an episode and pick the 3 best clip-worthy moments:

```bash
# Basic — transcribes with Whisper, then selects segments
clip-creator select episode.mp3

# Use OpenAI instead of Anthropic
clip-creator select episode.mp3 --llm-provider openai

# Save output to a file
clip-creator select episode.mp3 -o segments.json

# Skip transcription if you already have a transcript
clip-creator select episode.mp3 --transcript transcript.json
```

The transcript is saved automatically next to the audio file (e.g. `episode_transcript.json`) so you can reuse it later.

### Step 2: Cut clips

Take the segments from Step 1 and cut actual video clips with FFmpeg:

```bash
# Cut clips from a video using the segment output
clip-creator cut video.mp4 --segments segments.json --output-dir ./clips
```

This produces `clip_1.mp4`, `clip_2.mp4`, `clip_3.mp4` in the output directory.

The `--segments` flag accepts either the full JSON output from Step 1 or a bare list of segments.

### Backward compatibility

You can also run without a subcommand — it defaults to `select`:

```bash
clip-creator episode.mp3
```
