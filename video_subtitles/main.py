"""Batch processing script driven by central config."""

import argparse
import sys
import shutil
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- 경로 설정 ---
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

project_root = current_dir.parent.resolve()
sys.path.insert(1, str(project_root))

try:
    import config
except ImportError:
    sys.path.append(os.getcwd())
    import config

# 서비스 및 유틸리티 임포트
from services.audio_extractor import extract_audio
from services.transcriber import WhisperTranscriber
from services.translator import JapaneseToKoreanTranslator
from services.srt_writer import write_srt
from utils.segment_store import save_segments

class Logger:
    """Redirects stdout/stderr to both console and a log file."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure real-time saving

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_directories():
    """Create configured directories at project root."""
    Path(config.INPUT_DIR).mkdir(exist_ok=True)
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    Path(config.WORK_DIR).mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True) # Create logs directory

def get_video_files(input_dir: Path) -> List[Path]:
    extensions = [".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"]
    files = []
    for ext in extensions:
        files.extend(list(input_dir.glob(f"*{ext}")))
        files.extend(list(input_dir.glob(f"*{ext.upper()}")))
    return sorted(list(set(files)))

def backup_existing_file(file_path: Path):
    """If file exists, rename it with a timestamp to avoid overwriting."""
    if file_path.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_name(f"{file_path.stem}_{timestamp}{file_path.suffix}")
        file_path.rename(backup_path)
        return backup_path
    return None

def process_single_video(
    video_path: Path, 
    transcriber: WhisperTranscriber, 
    mode: str = "movie",
    skip_translate: bool = False
):
    start_time = time.time()
    print(f"\n[START] {'-'*15} Processing Start {'-'*15}")
    print(f"File   : {video_path.name}")
    print(f"Mode   : {mode.upper()}")
    print(f"{'-'*48}")

    work_dir = Path(config.WORK_DIR) / video_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Audio Extraction
    audio_path = extract_audio(video_path, work_dir)

    # Transcription
    segments = transcriber.transcribe(audio_path, mode=mode)
    if not segments:
        print(f"[WARN] No speech detected. Skipping.")
        return

    output_dir = Path(config.OUTPUT_DIR)
    ja_srt_path = output_dir / f"{video_path.stem}_{config.SOURCE_LANGUAGE}.srt"
    json_path = output_dir / f"{video_path.stem}_segments.json"
    
    # Backup & Save JA
    backup_ja = backup_existing_file(ja_srt_path)
    if backup_ja: print(f"[BACKUP] Created: {backup_ja.name}")
    
    write_srt(segments, ja_srt_path)
    save_segments(segments, json_path, source_video=video_path)
    print(f"[SUCCESS] Saved JA Subtitles: {ja_srt_path.name}")

    # Translation
    if not skip_translate:
        try:
            translator = JapaneseToKoreanTranslator()
            translated_segments = translator.translate_segments(segments)
            ko_srt_path = output_dir / f"{video_path.stem}_{config.TARGET_LANGUAGE}.srt"
            
            backup_ko = backup_existing_file(ko_srt_path)
            if backup_ko: print(f"[BACKUP] Created: {backup_ko.name}")
            
            write_srt(translated_segments, ko_srt_path)
            print(f"[SUCCESS] Saved KO Subtitles: {ko_srt_path.name}")
        except Exception as translate_e:
            print(f"\n[ERROR] Translation failed: {translate_e}")

    elapsed = time.time() - start_time
    print(f"{'-'*48}")
    print(f"[DONE] Processed: {video_path.name}")
    print(f"[REPORT] Time Taken: {elapsed:.2f}s | Segments: {len(segments)}")
    print(f"{'-'*48}")

    if config.CLEANUP_TEMP_FILES and work_dir.exists():
        shutil.rmtree(work_dir)

def main():
    parser = argparse.ArgumentParser(description="Automated Subtitle Generator")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, help="Whisper model size")
    parser.add_argument("--mode", type=str, default=config.DEFAULT_MODE, choices=["movie", "music"], help="Processing mode")
    parser.add_argument("--skip-translate", action="store_true", help="Skip translation")
    parser.add_argument("--file", type=str, help="Process a specific file")
    args = parser.parse_args()

    # Setup Logging
    setup_directories()
    log_name = f"session_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path = Path("logs") / log_name
    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout # Capture errors too

    print(f"\n{'='*50}")
    print(f"  WHISPER FASTER SUBTITLE GENERATOR v1.2.0")
    print(f"  Log File: {log_path}")
    print(f"{'='*50}")

    transcriber = WhisperTranscriber(model_name=args.model)

    if args.file:
        video_files = [Path(args.file)]
    else:
        video_files = get_video_files(Path(config.INPUT_DIR))
        if not video_files:
            print(f"[INFO] No video files found in '{config.INPUT_DIR}' folder.")
            return

    print(f"[INFO] Total Task: {len(video_files)} files found.")
    
    for i, video_path in enumerate(video_files, 1):
        try:
            print(f"\n[TASK {i}/{len(video_files)}]")
            process_single_video(video_path, transcriber, mode=args.mode, skip_translate=args.skip_translate)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Task {i} failed: {e}")
            continue

    print(f"\n{'='*50}")
    print(f"  ALL TASKS COMPLETED!")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
