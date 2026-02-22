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
    extract_audio,
    run_full_pipeline,
    step_detect_intro,
    step_select,
    step_transcribe,
)


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Add flags shared across subcommands."""
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
        help="Enable extra debug output (similarity curves, etc.)",
    )


def _collect_overrides(args: argparse.Namespace) -> dict[str, str] | None:
    """Build config overrides dict from CLI flags."""
    overrides: dict[str, str] = {}
    if getattr(args, "llm_provider", None):
        overrides["llm.provider"] = args.llm_provider
    if getattr(args, "llm_model", None):
        overrides["llm.model"] = args.llm_model
    if getattr(args, "whisper_mode", None):
        overrides["whisper.mode"] = args.whisper_mode
    if getattr(args, "whisper_model", None):
        overrides["whisper.model"] = args.whisper_model
    return overrides or None


def _load_config(args: argparse.Namespace):
    """Load config from args."""
    overrides = _collect_overrides(args)
    config_path = Path(args.config) if getattr(args, "config", None) else None
    return load_config(config_path=config_path, cli_overrides=overrides)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clip-creator",
        description="Extract the best short clips from a podcast episode.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- extract ---
    extract_p = subparsers.add_parser(
        "extract", help="Extract audio from a video file."
    )
    extract_p.add_argument("video", help="Path to the source video file")

    # --- detect-intro ---
    detect_p = subparsers.add_parser(
        "detect-intro", help="Detect intro/outro music boundaries."
    )
    detect_p.add_argument("audio", help="Path to the episode audio file")
    _add_common_flags(detect_p)

    # --- transcribe ---
    transcribe_p = subparsers.add_parser(
        "transcribe", help="Transcribe audio to text."
    )
    transcribe_p.add_argument("audio", help="Path to the episode audio file")
    transcribe_p.add_argument(
        "--boundaries",
        help="Path to boundaries JSON (auto-detected from {stem}_boundaries.json if not given)",
    )
    _add_common_flags(transcribe_p)

    # --- select ---
    select_p = subparsers.add_parser(
        "select", help="Select the best clip segments from a transcript."
    )
    select_p.add_argument("transcript", help="Path to the transcript JSON file")
    _add_common_flags(select_p)

    # --- cut ---
    cut_p = subparsers.add_parser(
        "cut", help="Cut video clips from segment selection output."
    )
    cut_p.add_argument("video", help="Path to the source video file")
    cut_p.add_argument(
        "--segments",
        required=True,
        help="Path to segments JSON (or inline JSON string)",
    )
    cut_p.add_argument(
        "--output-dir",
        default="./clips",
        help="Directory to write clips into (default: ./clips)",
    )

    # --- run ---
    run_p = subparsers.add_parser(
        "run", help="Full pipeline: video → audio → boundaries → transcribe → select → cut."
    )
    run_p.add_argument("video", help="Path to the source video file")
    run_p.add_argument(
        "--output-dir",
        default="./clips",
        help="Directory to write clips into (default: ./clips)",
    )
    _add_common_flags(run_p)

    return parser


# --- Handlers ---


def _handle_extract(args: argparse.Namespace) -> None:
    mp3_path = extract_audio(args.video)
    print(mp3_path)


def _handle_detect_intro(args: argparse.Namespace) -> None:
    config = _load_config(args)
    boundaries = step_detect_intro(args.audio, config)
    print(boundaries.model_dump_json(indent=2))


def _handle_transcribe(args: argparse.Namespace) -> None:
    config = _load_config(args)
    try:
        transcript = step_transcribe(
            args.audio, config, boundaries_path=args.boundaries
        )
    except TranscriptionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(transcript.model_dump_json(indent=2))


def _handle_select(args: argparse.Namespace) -> None:
    config = _load_config(args)
    try:
        segments = step_select(args.transcript, config)
    except LLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    # Output is already saved by step_select; print the segments list to stdout
    print(json.dumps(
        [s.model_dump() for s in segments],
        indent=2,
    ))


def _parse_segments(raw: str) -> list[CandidateSegment]:
    """Parse segments from a file path or inline JSON string."""
    path = Path(raw)
    if path.exists():
        data = json.loads(path.read_text())
    else:
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
    segments = _parse_segments(args.segments)
    try:
        results = cut_clips(args.video, segments, args.output_dir)
    except CutterError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(
        [r.model_dump() for r in results],
        indent=2,
    ))


def _handle_run(args: argparse.Namespace) -> None:
    config = _load_config(args)
    try:
        result = run_full_pipeline(args.video, config, args.output_dir)
    except (TranscriptionError, LLMError, CutterError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(result.model_dump_json(indent=2))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "extract": _handle_extract,
        "detect-intro": _handle_detect_intro,
        "transcribe": _handle_transcribe,
        "select": _handle_select,
        "cut": _handle_cut,
        "run": _handle_run,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
