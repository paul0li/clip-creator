"""Command-line interface for clip-creator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clip_creator.config import load_config
from clip_creator.models import LLMError, TranscriptionError
from clip_creator.pipeline import run_pipeline, run_pipeline_from_transcript


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clip-creator",
        description="Extract the best short clips from a podcast episode.",
    )
    parser.add_argument("audio", help="Path to the episode audio file")
    parser.add_argument(
        "--transcript",
        help="Path to a previously saved transcript JSON (skips transcription)",
    )
    parser.add_argument("--output", "-o", help="Write JSON output to this file instead of stdout")
    parser.add_argument("--config", help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--llm-provider", help='LLM provider: "anthropic" or "openai"')
    parser.add_argument("--llm-model", help="LLM model name")
    parser.add_argument("--whisper-mode", help='Whisper mode: "local" or "api"')
    parser.add_argument("--whisper-model", help="Whisper model size (tiny/base/small/medium/large-v3)")

    args = parser.parse_args()

    # Build CLI overrides dict
    overrides = {}
    if args.llm_provider:
        overrides["llm.provider"] = args.llm_provider
    if args.llm_model:
        overrides["llm.model"] = args.llm_model
    if args.whisper_mode:
        overrides["whisper.mode"] = args.whisper_mode
    if args.whisper_model:
        overrides["whisper.model"] = args.whisper_model

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path=config_path, cli_overrides=overrides or None)

    try:
        if args.transcript:
            result = run_pipeline_from_transcript(args.audio, args.transcript, config)
        else:
            result = run_pipeline(args.audio, config)
    except (TranscriptionError, LLMError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    json_output = result.model_dump_json(indent=2)

    if args.output:
        Path(args.output).write_text(json_output)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
