"""Whisper transcription service that assumes Japanese audio input."""

from typing import Any, Dict, List, Optional

import torch
from faster_whisper import WhisperModel
from tqdm.auto import tqdm

from .audio_chunker import AudioChunk


class WhisperTranscriber:
    """Thin wrapper around OpenAI Whisper for chunked transcription."""

    # Available model sizes include: "tiny", "base", "small", "medium", "large-v3".
    DEFAULT_MODEL_NAME = "medium"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print("CUDA acceleration enabled for Whisper transcription.")

        self.model = self._load_model(self.device)

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

            segments = self._transcribe_with_fallback(str(chunk.path))

            for segment in segments:
                start = chunk.start_time + float(segment.start)
                end = chunk.start_time + float(segment.end)
                text = segment.text.strip()
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

    def _load_model(self, device: str) -> WhisperModel:
        compute_type = "float16" if device == "cuda" else "int8"
        return WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute_type,
        )

    def _run_transcription(self, file_path: str):
        segments, _ = self.model.transcribe(
            file_path,
            language="ja",
            task="transcribe",
            beam_size=5,
        )
        return list(segments)

    def _transcribe_with_fallback(self, file_path: str):
        try:
            return self._run_transcription(file_path)
        except RuntimeError as exc:
            needs_fallback = self.device == "cuda" and "CUDNN_STATUS_NOT_INITIALIZED" in str(
                exc
            )
            if not needs_fallback:
                raise

            print("cuDNN initialization failed. Falling back to CPU for transcription.")
            self.device = "cpu"
            self.model = self._load_model(self.device)
            return self._run_transcription(file_path)
