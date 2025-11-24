"""Command-line entry point for generating subtitles from Japanese audio."""
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.audio_extractor import extract_audio
from services.audio_chunker import chunk_audio
from services.transcriber import WhisperTranscriber
from services.translator import JapaneseToKoreanTranslator
from services.srt_writer import write_srt
from utils.segment_store import load_segments, save_segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Japanese subtitles immediately and Korean subtitles either now or later.",
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        type=str,
        help="Path to the input mp4 video file to transcribe.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=WhisperTranscriber.DEFAULT_MODEL_NAME,
        help="Whisper model size to load (default: medium).",
    )
    parser.add_argument(
        "--translate-from",
        type=str,
        help="Translate a previously saved segments JSON file into Korean subtitles.",
    )
    parser.add_argument(
        "--korean-output",
        type=str,
        help="Optional override for the Korean SRT output path when translating.",
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Create only Japanese subtitles now and defer Korean translation.",
    )
    return parser


def run_transcription_pipeline(
    video_path: Path,
    model_name: str,
    *,
    translate: bool = True,
    translation_output: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Path, Path, Optional[Path]]:
    """
    Extract audio, run Whisper, emit Japanese subtitles, persist segments, and optionally translate.
    """
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    working_dir = video_path.parent / f"{video_path.stem}_work"
    chunks_dir = working_dir / "chunks"

    audio_path = extract_audio(video_path, working_dir)
    chunks = chunk_audio(audio_path, chunks_dir)

    if not chunks:
        raise RuntimeError("No audio chunks were generated.")

    transcriber = WhisperTranscriber(model_name=model_name)
    segments = transcriber.transcribe_chunks(chunks)
    if not segments:
        raise RuntimeError("No transcription segments were produced.")

    ja_output_path = video_path.with_name(f"{video_path.stem}_ja.srt")
    write_srt(segments, ja_output_path)

    segments_path = video_path.with_name(f"{video_path.stem}_segments.json")
    save_segments(segments, segments_path, source_video=video_path)
    ko_output_path: Optional[Path] = None
    if translate:
        ko_output_path = _write_korean_srt(
            segments,
            segments_file=segments_path,
            source_video=video_path,
            output_override=translation_output,
        )
    return segments, ja_output_path, segments_path, ko_output_path


def translate_saved_segments(
    segments_file: Path,
    *,
    output_override: Optional[Path] = None,
) -> Path:
    """
    Load saved segments and produce a Korean SRT translation.
    """
    file_path = segments_file.expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Segments file not found: {file_path}")

    segments, source_video = load_segments(file_path)
    if not segments:
        raise RuntimeError(f"No segments found in {file_path}")

    return _write_korean_srt(
        segments,
        segments_file=file_path,
        source_video=source_video,
        output_override=output_override,
    )


def _derive_korean_output(segments_file: Path, source_video: Optional[Path]) -> Path:
    """
    Decide on a reasonable default Korean subtitle filename.
    """
    if source_video:
        return source_video.with_name(f"{source_video.stem}_ko.srt")

    stem = segments_file.stem
    suffix = "_segments"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return segments_file.with_name(f"{stem}_ko.srt")


def _write_korean_srt(
    segments: List[Dict[str, Any]],
    *,
    segments_file: Path,
    source_video: Optional[Path],
    output_override: Optional[Path],
) -> Path:
    translator = JapaneseToKoreanTranslator()
    translated_segments = translator.translate_segments(segments)
    output_path = output_override or _derive_korean_output(segments_file, source_video)
    write_srt(translated_segments, output_path)
    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.translate_from:
        output_override = (
            Path(args.korean_output).expanduser().resolve() if args.korean_output else None
        )
        output_path = translate_saved_segments(
            Path(args.translate_from),
            output_override=output_override,
        )
        print(f"Korean subtitles saved to: {output_path}")
        return

    if not args.video_path:
        parser.error("You must provide a video path to transcribe or use --translate-from.")

    output_override = (
        Path(args.korean_output).expanduser().resolve() if args.korean_output else None
    )
    segments, ja_path, segments_path, ko_path = run_transcription_pipeline(
        Path(args.video_path),
        args.model,
        translate=not args.skip_translate,
        translation_output=output_override,
    )
    print(f"Japanese subtitles saved to: {ja_path}")
    print(f"Segments saved to: {segments_path}")
    if ko_path:
        print(f"Korean subtitles saved to: {ko_path}")
    else:
        print("Run again with --translate-from <segments_json> to generate Korean subtitles.")


if __name__ == "__main__":
    main()
