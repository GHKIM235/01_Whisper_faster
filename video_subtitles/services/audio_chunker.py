"""Split large audio files into smaller chunks to keep Whisper manageable."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# Adjustable chunk size in minutes to control how large each Whisper job becomes.
CHUNK_LENGTH_MINUTES = 5
CHUNK_DURATION_MS = CHUNK_LENGTH_MINUTES * 60 * 1000

# Silence trimming defaults.
MIN_SILENCE_LEN_MS = 700
ABSOLUTE_SILENCE_THRESH_DB = -50
THRESHOLD_PADDING_DB = 16


@dataclass
class AudioChunk:
    """Simple container for chunk path and absolute start offset in seconds."""

    path: Path
    start_time: float


def chunk_audio(audio_path: Path, chunk_dir: Path) -> List[AudioChunk]:
    """
    Slice the audio file into sequential WAV chunks after stripping silent ranges.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_file(audio_path)
    chunks: List[AudioChunk] = []

    voiced_spans = _detect_voiced_ranges(audio)
    chunk_index = 1

    for span_start, span_end in voiced_spans:
        current_ms = span_start
        while current_ms < span_end:
            chunk_end = min(current_ms + CHUNK_DURATION_MS, span_end)
            segment = audio[current_ms:chunk_end]
            duration = len(segment)
            if duration <= 0:
                current_ms = chunk_end
                continue

            chunk_path = chunk_dir / f"chunk_{chunk_index:04}.wav"
            if chunk_path.exists():
                chunk_path.unlink()
            segment.export(chunk_path, format="wav")
            chunks.append(AudioChunk(path=chunk_path, start_time=current_ms / 1000.0))

            chunk_index += 1
            current_ms = chunk_end

    return chunks


def _detect_voiced_ranges(audio: AudioSegment) -> List[Tuple[int, int]]:
    """
    Locate non-silent sections so we avoid wasting time transcribing empty audio.
    """
    if audio.dBFS == float("-inf"):
        silence_threshold = ABSOLUTE_SILENCE_THRESH_DB
    else:
        silence_threshold = max(audio.dBFS - THRESHOLD_PADDING_DB, ABSOLUTE_SILENCE_THRESH_DB)

    spans = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=silence_threshold,
    )

    if not spans:
        return [(0, len(audio))]
    return [(start, end) for start, end in spans]
