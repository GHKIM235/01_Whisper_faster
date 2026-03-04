"""Whisper transcription service optimized for direct file processing and GPU usage."""

import os
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional
from faster_whisper import WhisperModel
from tqdm.auto import tqdm

# Windows-specific DLL handling, only for Windows
if os.name == "nt":
    CUDA_DLL_HANDLE = None
    CUDA_DEFAULT_PATH = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
    CUDA_ENV_PATH = os.environ.get("CUDA_DLL_DIR")
    CUDA_PATH_CANDIDATES = [
        Path(CUDA_ENV_PATH) if CUDA_ENV_PATH else None,
        CUDA_DEFAULT_PATH,
    ]
    for candidate in CUDA_PATH_CANDIDATES:
        if candidate and candidate.exists():
            try:
                CUDA_DLL_HANDLE = os.add_dll_directory(str(candidate))
                break
            except Exception:
                pass


class WhisperTranscriber:
    """A wrapper around faster-whisper that handles full audio files efficiently."""

    DEFAULT_MODEL_NAME = "medium"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # In faster-whisper, float16 is standard for GPU, int8 for CPU.
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = self._load_model()

    def _load_model(self) -> WhisperModel:
        """Loads the Whisper model into the selected device."""
        return WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Transcribes the entire audio file using faster-whisper's built-in 
        long-form audio support and VAD filtering.
        """
        print(f"Starting transcription of: {audio_path.name}")
        
        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                language="ja",
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            results = []
            # We wrap the segments generator in tqdm to show progress.
            # info.duration gives the total length in seconds.
            with tqdm(total=round(info.duration), desc="Transcribing", unit="sec") as pbar:
                last_pos = 0
                for segment in segments:
                    results.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    })
                    # Update progress bar based on current timestamp
                    pbar.update(round(segment.end - last_pos))
                    last_pos = segment.end
                    
            return results
            
        except RuntimeError as e:
            if "CUDNN_STATUS_NOT_INITIALIZED" in str(e) and self.device == "cuda":
                print("\n[WARNING] cuDNN failure. Falling back to CPU...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = self._load_model()
                return self.transcribe(audio_path)
            raise e
