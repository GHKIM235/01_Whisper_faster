"""Reusable subtitle processing pipeline for CLI and GUI entry points."""

from __future__ import annotations

import io
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))
sys.path.insert(1, str(current_dir.parent.resolve()))

try:
    import config
except ImportError:
    sys.path.append(os.getcwd())
    import config

from services.audio_extractor import extract_audio
from services.gap_rescuer import GapRescuer
from services.srt_writer import write_srt
from services.transcriber import WhisperTranscriber
from services.translator import JapaneseToKoreanTranslator
from services.vocal_separator import separate_vocals
from utils.segment_store import load_segments, save_segments

LogCallback = Callable[[str], None]


@dataclass
class ProcessingOptions:
    model_name: str = config.MODEL_NAME
    mode: str = config.DEFAULT_MODE
    skip_translate: bool = config.SKIP_TRANSLATION
    translate_only: bool = False
    enable_rescue: bool = config.ENABLE_GAP_RESCUE


class TeeLogger(io.TextIOBase):
    """Mirrors stdout/stderr to the terminal, file, and optional callback."""

    def __init__(self, log_path: Path, callback: Optional[LogCallback] = None):
        self.terminal = sys.__stdout__
        self.log = open(log_path, "a", encoding="utf-8")
        self.callback = callback

    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        if self.callback and message:
            self.callback(message)
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        if not self.log.closed:
            self.log.close()


def setup_directories() -> None:
    output_dir = Path(config.OUTPUT_DIR)
    Path(config.INPUT_DIR).mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "jp").mkdir(exist_ok=True)
    (output_dir / "json").mkdir(exist_ok=True)
    Path(config.WORK_DIR).mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)


def get_video_files(input_dir: Path) -> List[Path]:
    extensions = [".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".wmv"]
    files: List[Path] = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*{ext}"))
        files.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def get_json_files(output_dir: Path) -> List[Path]:
    json_dir = output_dir / "json"
    files = list(json_dir.glob("*_segments.json")) if json_dir.exists() else []
    files.extend(output_dir.glob("*_segments.json"))
    return sorted(set(files))


def backup_existing_file(file_path: Path) -> Optional[Path]:
    if file_path.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_name(f"{file_path.stem}_{timestamp}{file_path.suffix}")
        shutil.move(str(file_path), str(backup_path))
        return backup_path
    return None


def collect_tasks(target: Optional[str], translate_only: bool) -> List[Path]:
    if target:
        target_path = Path(target)
        if target_path.is_dir():
            return get_json_files(target_path) if translate_only else get_video_files(target_path)
        return [target_path]

    if translate_only:
        return get_json_files(Path(config.OUTPUT_DIR))
    return get_video_files(Path(config.INPUT_DIR))


def create_transcriber_if_needed(options: ProcessingOptions) -> Optional[WhisperTranscriber]:
    if options.translate_only:
        return None
    return WhisperTranscriber(model_name=options.model_name)


def process_single_video(
    task_path: Path,
    transcriber: Optional[WhisperTranscriber],
    options: ProcessingOptions,
    log_callback: Optional[LogCallback] = None,
) -> None:
    start_time = time.time()

    if task_path.suffix.lower() == ".json":
        file_stem = task_path.name.replace("_segments.json", "")
        display_name = f"{file_stem} (from JSON)"
    else:
        file_stem = task_path.stem
        display_name = task_path.name

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"{file_stem}_session_{timestamp}.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger = TeeLogger(log_path, callback=log_callback)
    sys.stdout = logger
    sys.stderr = logger

    print(f"\n[START] {'-' * 15} Processing Start {'-' * 15}")
    print(f"Target : {display_name}")
    print(f"Mode   : {options.mode.upper()}")
    if options.translate_only:
        print("Type   : TRANSLATE-ONLY")
    print(f"{'-' * 48}")

    try:
        output_dir = Path(config.OUTPUT_DIR)
        json_path = output_dir / "json" / f"{file_stem}_segments.json"
        segments = []

        if options.translate_only:
            if task_path.suffix.lower() == ".json" and task_path.exists():
                json_path = task_path
            elif not json_path.exists():
                json_path = output_dir / f"{file_stem}_segments.json"

            if not json_path.exists():
                print(f"[ERROR] Segments file not found: {json_path.name}")
                return

            print(f"[LOAD] Loading segments from: {json_path.name}")
            segments, _ = load_segments(json_path)
        else:
            work_dir = Path(config.WORK_DIR) / file_stem
            work_dir.mkdir(parents=True, exist_ok=True)

            if not task_path.exists():
                print(f"[ERROR] Video file not found: {task_path}")
                return

            audio_path = extract_audio(task_path, work_dir)
            audio_path = separate_vocals(audio_path, work_dir)

            if not transcriber:
                print("[ERROR] Transcriber not initialized.")
                return

            segments = transcriber.transcribe(audio_path, mode=options.mode)
            if not segments:
                print("[WARN] No speech detected. Skipping.")
                return

            jp_srt_path = output_dir / "jp" / f"{file_stem}_jp.srt"
            backup_jp = backup_existing_file(jp_srt_path)
            if backup_jp:
                print(f"[BACKUP] Created: {backup_jp.name}")

            write_srt(segments, jp_srt_path)
            save_segments(segments, json_path, source_video=task_path)
            print(f"[SUCCESS] Saved JP Subtitles: {jp_srt_path.name}")

            if options.enable_rescue:
                backup_json = backup_existing_file(json_path)
                if backup_json:
                    print(f"[BACKUP] JSON saved: {backup_json.name}")

                rescuer = GapRescuer(transcriber)
                rescued_segments = rescuer.rescue(segments, audio_path)
                if len(rescued_segments) > len(segments):
                    segments = rescued_segments
                    write_srt(segments, jp_srt_path)
                    save_segments(segments, json_path, source_video=task_path)
                    print("[SUCCESS] Updated JP Subtitles with rescued lines.")

        deepl_usage = "N/A"
        if (not options.skip_translate or options.translate_only) and segments:
            try:
                print("[RUN] Starting Translation...")
                translator = JapaneseToKoreanTranslator()
                translated_segments = translator.translate_segments(segments)
                ko_srt_path = output_dir / f"{file_stem}.srt"

                backup_ko = backup_existing_file(ko_srt_path)
                if backup_ko:
                    print(f"[BACKUP] Created: {backup_ko.name}")

                write_srt(translated_segments, ko_srt_path)
                print(f"[SUCCESS] Saved KO Subtitles: {ko_srt_path.name}")
                deepl_usage = translator.get_usage()
            except Exception as exc:
                print(f"\n[ERROR] Translation failed: {exc}")

        elapsed = time.time() - start_time
        print(f"{'-' * 48}")
        print(f"[DONE] Processed: {file_stem}")
        print(f"[REPORT] Time Taken : {elapsed:.2f}s")
        print(f"[REPORT] Segments   : {len(segments)}")
        if not options.skip_translate or options.translate_only:
            print(f"[REPORT] DeepL Usage: {deepl_usage}")
        print(f"{'-' * 48}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.close()


def run_batch(
    tasks: Iterable[Path],
    options: ProcessingOptions,
    log_callback: Optional[LogCallback] = None,
) -> None:
    task_list = list(tasks)
    if not task_list:
        raise ValueError("No target files found.")

    setup_directories()
    transcriber = create_transcriber_if_needed(options)

    print(f"\n{'=' * 50}")
    print("  WHISPER FASTER SUBTITLE GENERATOR")
    if options.translate_only:
        print("  [STATUS] Mode: TRANSLATE-ONLY")
    elif options.skip_translate:
        print("  [STATUS] Translation: DISABLED")
    else:
        print("  [STATUS] Translation: ENABLED")
    if options.enable_rescue:
        print("  [STATUS] Gap Rescue : ENABLED")
    print(f"{'=' * 50}")
    print(f"[INFO] Total Task: {len(task_list)} targets found.")

    for index, task_path in enumerate(task_list, start=1):
        try:
            print(f"\n[TASK {index}/{len(task_list)}] Processing: {task_path.name}")
            process_single_video(task_path, transcriber, options, log_callback=log_callback)
        except Exception as exc:
            print(f"\n[CRITICAL ERROR] Task {index} failed: {exc}")

    print(f"\n{'=' * 50}")
    print("  ALL TASKS COMPLETED!")
    print(f"{'=' * 50}\n")
