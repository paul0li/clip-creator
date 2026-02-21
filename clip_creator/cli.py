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
from clip_creator.pipeline import (
    run_full_pipeline,
    run_pipeline,
    run_pipeline_from_transcript,
)


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Add flags shared by select and run subcommands."""
    parser.add_argument(
        "--config", help="Path to config.yaml (default: ./config.yaml)"
    )
    parser.add_argument("--llm-provider", help='LLM provider: "anthropic" or "openai"')
    parser.add_argument("--llm-model", help="LLM model name")
    parser.add_argument("--whisper-mode", help='Whisper mode: "local" or "api"')
    parser.add_argument(
        "--whisper-model",
        help="Whisper model size (tiny/base/small/medium/large-v3)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate files (transcript, segments) for inspection",
    )


def _collect_overrides(args: argparse.Namespace) -> dict[str, str] | None:
    """Build config overrides dict from CLI flags."""
    overrides: dict[str, str] = {}
    if args.llm_provider:
        overrides["llm.provider"] = args.llm_provider
    if args.llm_model:
        overrides["llm.model"] = args.llm_model
    if args.whisper_mode:
        overrides["whisper.mode"] = args.whisper_mode
    if args.whisper_model:
        overrides["whisper.model"] = args.whisper_model
    return overrides or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clip-creator",
        description="Extract the best short clips from a podcast episode.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- select subcommand ---
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
    _add_common_flags(select_parser)

    # --- cut subcommand ---
    cut_parser = subparsers.add_parser(
        "cut", help="Cut video clips from segment selection output."
    )
    cut_parser.add_argument("video", help="Path to the source video file")
    cut_parser.add_argument(
        "--segments",
        required=True,
        help="Path to segments JSON output (or inline JSON string)",
    )
    cut_parser.add_argument(
        "--output-dir",
        default="./clips",
        help="Directory to write clips into (default: ./clips)",
    )

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run", help="Full pipeline: video → segments → clips."
    )
    run_parser.add_argument("video", help="Path to the source video file")
    run_parser.add_argument(
        "--output-dir",
        default="./clips",
        help="Directory to write clips into (default: ./clips)",
    )
    _add_common_flags(run_parser)

    return parser


def _handle_select(args: argparse.Namespace) -> None:
    """Run the segment-selection pipeline."""
    overrides = _collect_overrides(args)
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path=config_path, cli_overrides=overrides)

    try:
        if args.transcript:
            result = run_pipeline_from_transcript(args.audio, args.transcript, config)
        else:
            result = run_pipeline(args.audio, config, debug=args.debug)
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
    """Run the clip-cutting pipeline."""
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


def _handle_run(args: argparse.Namespace) -> None:
    """Run the full pipeline: video → audio → segments → clips."""
    overrides = _collect_overrides(args)
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path=config_path, cli_overrides=overrides)

    try:
        result = run_full_pipeline(
            args.video, config, args.output_dir, debug=args.debug
        )
    except (TranscriptionError, LLMError, CutterError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(result.model_dump_json(indent=2))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "select":
        _handle_select(args)
    elif args.command == "cut":
        _handle_cut(args)
    elif args.command == "run":
        _handle_run(args)
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
