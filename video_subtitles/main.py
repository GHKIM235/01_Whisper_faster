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
from utils.segment_store import save_segments, load_segments

class Logger:
    """Redirects stdout/stderr to both console and a log file."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if self.log:
            self.log.close()

def setup_directories():
    """Create configured directories at project root."""
    output_dir = Path(config.OUTPUT_DIR)
    Path(config.INPUT_DIR).mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "jp").mkdir(exist_ok=True)
    (output_dir / "json").mkdir(exist_ok=True)
    Path(config.WORK_DIR).mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

def get_video_files(input_dir: Path) -> List[Path]:
    extensions = [".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"]
    files = []
    for ext in extensions:
        files.extend(list(input_dir.glob(f"*{ext}")))
        files.extend(list(input_dir.glob(f"*{ext.upper()}")))
    return sorted(list(set(files)))

def get_json_files(output_dir: Path) -> List[Path]:
    """Find all segment JSON files in the output directory (including subfolders)."""
    json_dir = output_dir / "json"
    files = list(json_dir.glob("*_segments.json")) if json_dir.exists() else []
    # 루트 폴더에서도 찾아봄 (하위 호환성)
    files.extend(list(output_dir.glob("*_segments.json")))
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
    task_path: Path, 
    transcriber: Optional[WhisperTranscriber], 
    mode: str = "movie",
    skip_translate: bool = False,
    translate_only: bool = False
):
    start_time = time.time()
    
    # Handle filename logic (whether task_path is video or json)
    if task_path.suffix.lower() == ".json":
        file_stem = task_path.name.replace("_segments.json", "")
        display_name = f"{file_stem} (from JSON)"
    else:
        file_stem = task_path.stem
        display_name = task_path.name

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f"{file_stem}_session_{timestamp}.log"
    log_path = Path("logs") / log_name
    
    # Initialize logger
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    video_logger = Logger(log_path)
    sys.stdout = video_logger
    sys.stderr = video_logger

    print(f"\n[START] {'-'*15} Processing Start {'-'*15}")
    print(f"Target : {display_name}")
    print(f"Mode   : {mode.upper()}")
    if translate_only: print(f"Type   : TRANSLATE-ONLY")
    print(f"{'-'*48}")

    try:
        output_dir = Path(config.OUTPUT_DIR)
        json_path = output_dir / "json" / f"{file_stem}_segments.json"
        segments = []

        if translate_only:
            # --- TRANSLATE ONLY MODE ---
            if not json_path.exists():
                # 루트 폴더에서도 찾아봄 (하위 호환성)
                json_path = output_dir / f"{file_stem}_segments.json"
                
            if not json_path.exists():
                print(f"[ERROR] Segments file not found: {json_path.name}")
                return
            
            print(f"[LOAD] Loading segments from: {json_path.name}")
            segments, _ = load_segments(json_path)
        else:
            # --- NORMAL MODE (Extraction + Whisper) ---
            work_dir = Path(config.WORK_DIR) / file_stem
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # Audio Extraction (Requires video to exist)
            if not task_path.exists():
                print(f"[ERROR] Video file not found: {task_path}")
                return
                
            audio_path = extract_audio(task_path, work_dir)

            # Transcription
            if not transcriber:
                print("[ERROR] Transcriber not initialized.")
                return
            segments = transcriber.transcribe(audio_path, mode=mode)
            
            if not segments:
                print(f"[WARN] No speech detected. Skipping.")
                return

            # Save Japanese SRT & Raw JSON
            jp_srt_path = output_dir / "jp" / f"{file_stem}_jp.srt"
            backup_jp = backup_existing_file(jp_srt_path)
            if backup_jp: print(f"[BACKUP] Created: {backup_jp.name}")
            
            write_srt(segments, jp_srt_path)
            save_segments(segments, json_path, source_video=task_path)
            print(f"[SUCCESS] Saved JP Subtitles: {jp_srt_path.name}")

        # --- Translation Step ---
        deepl_usage = "N/A"
        if (not skip_translate or translate_only) and segments:
            try:
                print(f"[RUN] Starting Translation...")
                translator = JapaneseToKoreanTranslator()
                translated_segments = translator.translate_segments(segments)
                ko_srt_path = output_dir / f"{file_stem}.srt"
                
                backup_ko = backup_existing_file(ko_srt_path)
                if backup_ko: print(f"[BACKUP] Created: {backup_ko.name}")
                
                write_srt(translated_segments, ko_srt_path)
                print(f"[SUCCESS] Saved KO Subtitles: {ko_srt_path.name}")
                
                deepl_usage = translator.get_usage()
            except Exception as translate_e:
                print(f"\n[ERROR] Translation failed: {translate_e}")

        elapsed = time.time() - start_time
        print(f"{'-'*48}")
        print(f"[DONE] Processed: {file_stem}")
        print(f"[REPORT] Time Taken : {elapsed:.2f}s")
        print(f"[REPORT] Segments   : {len(segments)}")
        if not skip_translate or translate_only:
            print(f"[REPORT] DeepL Usage: {deepl_usage}")
        print(f"{'-'*48}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        video_logger.close()

def main():
    parser = argparse.ArgumentParser(description="Automated Subtitle Generator")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, help="Whisper model size")
    parser.add_argument("--mode", type=str, default=config.DEFAULT_MODE, choices=["movie", "music"], help="Processing mode")
    parser.add_argument("--skip-translate", action="store_true", default=config.SKIP_TRANSLATION, help="Skip translation")
    parser.add_argument("--translate-only", action="store_true", help="Translate existing segments JSON without transcription")
    parser.add_argument("--file", type=str, help="Process a specific video or JSON file")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  WHISPER FASTER SUBTITLE GENERATOR v1.2.0")
    if args.translate_only:
        print(f"  [STATUS] Mode: TRANSLATE-ONLY (GPU Analysis Skipped)")
    elif args.skip_translate:
        print(f"  [STATUS] Translation: DISABLED (API Savings Mode)")
    else:
        print(f"  [STATUS] Translation: ENABLED")
    print(f"{'='*50}")

    setup_directories()
    
    # Initialize transcriber only if needed
    transcriber = None
    if not args.translate_only:
        transcriber = WhisperTranscriber(model_name=args.model)

    # Resolve task list
    task_files = []
    if args.file:
        p = Path(args.file)
        if p.is_dir():
            # 폴더 경로가 입력된 경우 해당 폴더 내의 영상 스캔
            task_files = get_video_files(p)
            if not task_files:
                print(f"[INFO] No video files found in directory: {p}")
                return
            print(f"[INFO] Directory detected. Found {len(task_files)} videos in: {p.name}")
        else:
            task_files = [p]
    else:
        if args.translate_only:
            # Scan output folder for JSONs
            task_files = get_json_files(Path(config.OUTPUT_DIR))
            if not task_files:
                print(f"[INFO] No segment JSON files found in '{config.OUTPUT_DIR}'.")
                return
        else:
            # Scan input folder for Videos
            task_files = get_video_files(Path(config.INPUT_DIR))
            if not task_files:
                print(f"[INFO] No video files found in '{config.INPUT_DIR}'.")
                return

    print(f"[INFO] Total Task: {len(task_files)} targets found.")
    
    for i, task_path in enumerate(task_files, 1):
        try:
            print(f"\n[TASK {i}/{len(task_files)}] Processing: {task_path.name}")
            process_single_video(
                task_path, 
                transcriber, 
                mode=args.mode, 
                skip_translate=args.skip_translate,
                translate_only=args.translate_only
            )
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Task {i} failed: {e}")
            continue

    print(f"\n{'='*50}")
    print(f"  ALL TASKS COMPLETED!")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
