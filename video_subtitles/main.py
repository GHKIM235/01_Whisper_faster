"""Batch processing script driven by central config."""

import argparse
import sys
import shutil
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to sys.path so we can import config.py
# This resolves "config" import error in IDEs and when running from subfolder
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    # If still not found, search in current directory as fallback
    sys.path.append(str(Path(__file__).parent))
    import config

from services.audio_extractor import extract_audio
from services.transcriber import WhisperTranscriber
from services.translator import JapaneseToKoreanTranslator
from services.srt_writer import write_srt
from utils.segment_store import save_segments

def setup_directories():
    """Create configured directories."""
    Path(config.INPUT_DIR).mkdir(exist_ok=True)
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    Path(config.WORK_DIR).mkdir(exist_ok=True)

def get_video_files(input_dir: Path) -> List[Path]:
    extensions = [".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"]
    files = []
    for ext in extensions:
        files.extend(list(input_dir.glob(f"*{ext}")))
        files.extend(list(input_dir.glob(f"*{ext.upper()}")))
    return sorted(list(set(files)))

def process_single_video(
    video_path: Path, 
    transcriber: WhisperTranscriber, 
    skip_translate: bool = False
):
    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*60}")

    work_dir = Path(config.WORK_DIR) / video_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = extract_audio(video_path, work_dir)

    segments = transcriber.transcribe(audio_path)
    if not segments:
        print(f"No speech detected. Skipping.")
        return

    output_dir = Path(config.OUTPUT_DIR)
    ja_srt_path = output_dir / f"{video_path.stem}_{config.SOURCE_LANGUAGE}.srt"
    json_path = output_dir / f"{video_path.stem}_segments.json"
    
    write_srt(segments, ja_srt_path)
    save_segments(segments, json_path, source_video=video_path)
    print(f"Saved {config.SOURCE_LANGUAGE.upper()} subtitles: {ja_srt_path.name}")

    if not skip_translate:
        translator = JapaneseToKoreanTranslator()
        translated_segments = translator.translate_segments(segments)
        ko_srt_path = output_dir / f"{video_path.stem}_{config.TARGET_LANGUAGE}.srt"
        write_srt(translated_segments, ko_srt_path)
        print(f"Saved {config.TARGET_LANGUAGE.upper()} subtitles: {ko_srt_path.name}")

    # Cleanup temp audio if configured
    if config.CLEANUP_TEMP_FILES and work_dir.exists():
        print(f"Cleaning up temp files in: {work_dir}")
        shutil.rmtree(work_dir)

def main():
    parser = argparse.ArgumentParser(description="Automated Subtitle Generator")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, help="Whisper model size")
    parser.add_argument("--skip-translate", action="store_true", help="Skip translation")
    parser.add_argument("--file", type=str, help="Process a specific file")
    args = parser.parse_args()

    setup_directories()
    transcriber = WhisperTranscriber(model_name=args.model)

    if args.file:
        video_files = [Path(args.file)]
    else:
        video_files = get_video_files(Path(config.INPUT_DIR))
        if not video_files:
            print(f"No video files found in '{config.INPUT_DIR}' folder.")
            return

    print(f"Found {len(video_files)} files to process.")
    for i, video_path in enumerate(video_files, 1):
        try:
            print(f"\n[Progress: {i}/{len(video_files)}]")
            process_single_video(video_path, transcriber, args.skip_translate)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed to process {video_path.name}: {e}")
            continue

    print(f"\n{'='*60}\nAll tasks completed!\n{'='*60}")

if __name__ == "__main__":
    main()
