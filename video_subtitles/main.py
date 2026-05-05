"""CLI entry point for the subtitle generator."""

import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))
sys.path.insert(1, str(current_dir.parent.resolve()))

import config
from app_pipeline import ProcessingOptions, collect_tasks, run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automated Subtitle Generator")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, help="Whisper model size")
    parser.add_argument(
        "--mode",
        type=str,
        default=config.DEFAULT_MODE,
        choices=["movie", "music"],
        help="Processing mode",
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        default=config.SKIP_TRANSLATION,
        help="Skip translation",
    )
    parser.add_argument(
        "--translate-only",
        action="store_true",
        help="Translate existing segments JSON without transcription",
    )
    parser.add_argument(
        "--rescue",
        action="store_true",
        default=config.ENABLE_GAP_RESCUE,
        help="Enable high-recall gap rescue analysis",
    )
    parser.add_argument(
        "target",
        type=str,
        nargs="?",
        help="Specific video, JSON, or directory to process (default: configured folder)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    options = ProcessingOptions(
        model_name=args.model,
        mode=args.mode,
        skip_translate=args.skip_translate,
        translate_only=args.translate_only,
        enable_rescue=args.rescue,
    )
    tasks = collect_tasks(args.target, args.translate_only)
    if not tasks:
        if args.translate_only:
            print(f"[INFO] No segment JSON files found in '{config.OUTPUT_DIR}'.")
        else:
            target_dir = args.target or config.INPUT_DIR
            print(f"[INFO] No video files found in '{target_dir}'.")
        return
    run_batch(tasks, options)


if __name__ == "__main__":
    main()
