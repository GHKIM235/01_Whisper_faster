"""Whisper transcription service that assumes Japanese audio input."""

from typing import Any, Dict, List, Optional

import torch
import whisper
from tqdm.auto import tqdm

from .audio_chunker import AudioChunk


class WhisperTranscriber:
    """Thin wrapper around OpenAI Whisper for chunked transcription."""

    DEFAULT_MODEL_NAME = "medium"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print("CUDA acceleration enabled for Whisper transcription.")
        self.model = whisper.load_model(self.model_name, device=self.device)

    def transcribe_chunks(self, chunks: List[AudioChunk]) -> List[Dict[str, Any]]:
        """
        Run transcription for each chunk and normalize timestamps back to the
        original audio timeline.
        """
        results: List[Dict[str, Any]] = []

        if not chunks:
            return results

        total_chunks = len(chunks)
        progress_bar = tqdm(
            chunks,
            total=total_chunks,
            desc="Transcribing chunks",
            unit="chunk",
        )

        for index, chunk in enumerate(progress_bar, start=1):
            progress_bar.set_postfix_str(f"chunk {index}/{total_chunks}")

            transcription = self.model.transcribe(
                str(chunk.path),
                language="ja",
                task="transcribe",
                fp16=self.device == "cuda",
            )

            for segment in transcription.get("segments", []):
                start = chunk.start_time + float(segment["start"])
                end = chunk.start_time + float(segment["end"])
                text = segment["text"].strip()
                if not text:
                    continue
                results.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text,
                    }
                )

        progress_bar.close()
        return results
