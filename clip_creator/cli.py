"""Command-line interface for clip-creator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clip_creator.config import load_config
from clip_creator.cutter import cut_clips
from clip_creator.models import (
    CandidateSegment,
    CutterError,
    LLMError,
    TranscriptionError,
)
from clip_creator.pipeline import run_pipeline, run_pipeline_from_transcript


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clip-creator",
        description="Extract the best short clips from a podcast episode.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- select subcommand (Card 1) ---
    select_parser = subparsers.add_parser(
        "select", help="Analyse audio and select the best clip segments."
    )
    select_parser.add_argument("audio", help="Path to the episode audio file")
    select_parser.add_argument(
        "--transcript",
        help="Path to a previously saved transcript JSON (skips transcription)",
    )
    select_parser.add_argument(
        "--output", "-o", help="Write JSON output to this file instead of stdout"
    )
    select_parser.add_argument(
        "--config", help="Path to config.yaml (default: ./config.yaml)"
    )
    select_parser.add_argument("--llm-provider", help='LLM provider: "anthropic" or "openai"')
    select_parser.add_argument("--llm-model", help="LLM model name")
    select_parser.add_argument("--whisper-mode", help='Whisper mode: "local" or "api"')
    select_parser.add_argument(
        "--whisper-model",
        help="Whisper model size (tiny/base/small/medium/large-v3)",
    )

    # --- cut subcommand (Card 2) ---
    cut_parser = subparsers.add_parser(
        "cut", help="Cut video clips from Card 1 segment output."
    )
    cut_parser.add_argument("video", help="Path to the source video file")
    cut_parser.add_argument(
        "--segments",
        required=True,
        help="Path to Card 1 JSON output (or inline JSON string)",
    )
    cut_parser.add_argument(
        "--output-dir",
        default="./clips",
        help="Directory to write clips into (default: ./clips)",
    )

    return parser


def _handle_select(args: argparse.Namespace) -> None:
    """Run the segment-selection pipeline (Card 1)."""
    overrides: dict[str, str] = {}
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


def _parse_segments(raw: str) -> list[CandidateSegment]:
    """Parse segments from a file path or inline JSON string."""
    path = Path(raw)
    if path.exists():
        data = json.loads(path.read_text())
    else:
        # Try parsing as inline JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Error: '{raw}' is not a valid file path or JSON string.", file=sys.stderr)
            sys.exit(1)

    # Accept either the full PipelineOutput (has .segments) or a bare list
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]

    if not isinstance(data, list):
        print("Error: segments JSON must be a list or an object with a 'segments' key.", file=sys.stderr)
        sys.exit(1)

    return [CandidateSegment(**s) for s in data]


def _handle_cut(args: argparse.Namespace) -> None:
    """Run the clip-cutting pipeline (Card 2)."""
    segments = _parse_segments(args.segments)

    try:
        results = cut_clips(args.video, segments, args.output_dir)
    except CutterError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    json_output = json.dumps(
        [r.model_dump() for r in results],
        indent=2,
    )
    print(json_output)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "select":
        _handle_select(args)
    elif args.command == "cut":
        _handle_cut(args)
    else:
        # Backward compat: bare `clip-creator audio.mp3` (no subcommand)
        # Re-parse with the first positional as audio for select
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            sys.argv.insert(1, "select")
            args = parser.parse_args()
            _handle_select(args)
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
