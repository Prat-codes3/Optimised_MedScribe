"""
Optimised MedScribe — CLI Entry Point

This is the file you run from the terminal.  It is intentionally THIN —
its only jobs are:
    1. Parse command-line arguments (what the user typed).
    2. Override the default AppConfig with those arguments.
    3. Hand everything off to the appropriate pipeline.
    4. Print the final result.

All the actual intelligence lives in the pipeline and core modules.
This file contains ZERO business logic, ZERO model code, and ZERO
file processing code.

WHY KEEP main.py THIN?
    In the old MedScribe v1.0, main.py was 150+ lines containing CLI parsing,
    model loading, transcription, diarization, merging, and saving — all
    tangled together.  If you wanted to add a server, you'd have to duplicate
    all of that.

    Now, main.py is just a "remote control" — it reads what buttons you press
    and tells the pipeline to do the work.  The server (server.py) will use
    the exact same pipeline without duplicating a single line of code.

USAGE EXAMPLES:
    # Basic transcription (uses defaults from config.py and .env)
    python main.py --file recording.wav

    # With speaker identification
    python main.py --file recording.wav --diarize

    # JSON output for SOAP notes
    python main.py --file recording.wav --diarize --output json

    # Process an entire folder
    python main.py --dir D:/recordings/ --diarize --output json

    # Override model and device
    python main.py --file recording.wav --model large-v3 --device cuda

    # Live transcription (Phase 2 — to be added later)
    python main.py --live
"""

import argparse
import sys

from config import AppConfig
from pipelines.batch_pipeline import BatchPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all CLI flags.

    This function is separated from main() so that:
        - We can test the parser independently.
        - The --help output is generated cleanly.
        - Adding new flags is easy — just add them here.

    Returns:
        A configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="medscribe",
        description=(
            "Optimised MedScribe v2.0 — Medical Audio Transcription\n"
            "Transcribe audio files with optional speaker identification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --file consult.wav\n"
            "  python main.py --file consult.wav --diarize --output json\n"
            "  python main.py --dir D:/recordings/ --diarize\n"
        ),
    )

    # ── Input Source (mutually exclusive) ──────────────────────────────
    # The user must provide EITHER --file OR --dir, but not both.
    # In Phase 2, --live will be added as a third option.
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a single audio file to transcribe.",
    )

    input_group.add_argument(
        "--dir", "-d",
        type=str,
        help="Path to a directory of audio files to batch-process.",
    )

    # ── Model Settings ────────────────────────────────────────────────
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,  # None means "use config default"
        help=(
            "Whisper model size. Options: tiny, base, small, medium, "
            "large-v3, distil-large-v3. Default: distil-large-v3"
        ),
    )

    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Language code for transcription (e.g., 'hi', 'en'). Default: hi",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size for decoding. Higher = more accurate but slower. Default: 5",
    )

    # ── Device Settings ───────────────────────────────────────────────
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Compute device. 'auto' picks GPU if available. Default: auto",
    )

    parser.add_argument(
        "--compute-type",
        choices=["auto", "float16", "int8_float16", "int8"],
        default=None,
        help="Numeric precision. 'auto' picks the best for your device. Default: auto",
    )

    # ── Pipeline Settings ─────────────────────────────────────────────
    parser.add_argument(
        "--diarize",
        action="store_true",
        default=False,
        help="Enable speaker identification (requires HF_TOKEN in .env).",
    )

    parser.add_argument(
        "--output", "-o",
        choices=["txt", "json"],
        default=None,
        help="Output format. 'json' is structured for SOAP notes. Default: txt",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of 30s audio windows processed per GPU call. Default: 8",
    )

    return parser


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply CLI arguments on top of the default AppConfig.

    Only values explicitly provided by the user are overridden.
    If the user didn't pass a flag, the config keeps its default
    (from config.py or .env file).

    This implements the priority chain:
        CLI argument (highest) > .env variable > config.py default (lowest)

    Args:
        config: The default AppConfig instance.
        args: Parsed CLI arguments from argparse.

    Returns:
        The same AppConfig instance with overrides applied.
    """
    # Only override if the user explicitly passed the flag.
    # We check for None because that's the default we set in argparse.
    if args.model is not None:
        config.model_size = args.model

    if args.language is not None:
        config.language = args.language

    if args.beam_size is not None:
        config.beam_size = args.beam_size

    if args.device is not None:
        config.device_preference = args.device
        config.resolve()  # Re-resolve device and compute type

    if args.compute_type is not None:
        config.compute_type_preference = args.compute_type
        config.resolve()

    if args.diarize:
        config.diarize = True

    if args.output is not None:
        config.output_format = args.output

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    return config


def main():
    """Main entry point — parse arguments and run the pipeline."""

    # ── Step 1: Parse CLI arguments ───────────────────────────────────
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Step 2: Create config with defaults, then apply CLI overrides ─
    config = AppConfig()
    config = apply_cli_overrides(config, args)

    # ── Step 3: Print the active configuration ────────────────────────
    print(config.summary())

    # ── Step 4: Create and run the pipeline ───────────────────────────
    pipeline = BatchPipeline(config)

    if args.file:
        # Single file mode
        result = pipeline.process_file(args.file)

        # Exit with error code if processing failed
        if not result["success"]:
            sys.exit(1)

    elif args.dir:
        # Directory batch mode
        results = pipeline.process_directory(args.dir)

        # Exit with error code if ANY file failed
        failed = [r for r in results if not r["success"]]
        if failed:
            print(f"\n{len(failed)} file(s) had errors.")
            sys.exit(1)


if __name__ == "__main__":
    main()
