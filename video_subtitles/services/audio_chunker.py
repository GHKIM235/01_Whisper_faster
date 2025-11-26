"""Split large audio files into smaller chunks to keep Whisper manageable."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pydub import AudioSegment

# Adjustable chunk size in minutes to control how large each Whisper job becomes.
CHUNK_LENGTH_MINUTES = 5
CHUNK_DURATION_MS = CHUNK_LENGTH_MINUTES * 60 * 1000


@dataclass
class AudioChunk:
    """Simple container for chunk path and absolute start offset in seconds."""

    path: Path
    start_time: float


def chunk_audio(audio_path: Path, chunk_dir: Path) -> List[AudioChunk]:
    """
    Slice the audio file into sequential WAV chunks.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_file(audio_path)
    chunks: List[AudioChunk] = []

    for index, start_ms in enumerate(range(0, len(audio), CHUNK_DURATION_MS), start=1):
        segment = audio[start_ms : start_ms + CHUNK_DURATION_MS]
        chunk_path = chunk_dir / f"chunk_{index:04}.wav"
        if chunk_path.exists():
            chunk_path.unlink()
        segment.export(chunk_path, format="wav")
        chunks.append(AudioChunk(path=chunk_path, start_time=start_ms / 1000.0))

    return chunks

