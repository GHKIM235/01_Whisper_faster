"""Whisper transcription service optimized with hardware control settings."""

import os
import torch
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from faster_whisper import WhisperModel
from tqdm.auto import tqdm

# Import central config
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    import config
except ImportError:
    class ConfigMock:
        MODEL_NAME = "medium"
        SOURCE_LANGUAGE = "ja"
        USE_VAD_FILTER = True
        DEVICE = "auto"
        COMPUTE_TYPE = "auto"
    config = ConfigMock()

# Windows-specific DLL handling
if os.name == "nt":
    CUDA_DLL_HANDLE = None
    CUDA_DEFAULT_PATH = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
    CUDA_ENV_PATH = os.environ.get("CUDA_DLL_DIR")
    CUDA_PATH_CANDIDATES = [Path(CUDA_ENV_PATH) if CUDA_ENV_PATH else None, CUDA_DEFAULT_PATH]
    for candidate in CUDA_PATH_CANDIDATES:
        if candidate and candidate.exists():
            try:
                CUDA_DLL_HANDLE = os.add_dll_directory(str(candidate))
                break
            except Exception: pass

class WhisperTranscriber:
    """A wrapper around faster-whisper with hardware-aware transcription."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.MODEL_NAME
        
        # Decide on device (GPU/CPU)
        if config.DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.DEVICE
            
        # Decide on compute_type
        if config.COMPUTE_TYPE == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = config.COMPUTE_TYPE
            
        print(f"Using device: {self.device} | Compute type: {self.compute_type} | Model: {self.model_name}")
        self.model = self._load_model()

    def _load_model(self) -> WhisperModel:
        return WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: Path) -> List[Dict[str, Any]]:
        print(f"Starting transcription: {audio_path.name}")
        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                language=config.SOURCE_LANGUAGE,
                task="transcribe",
                beam_size=5,
                vad_filter=config.USE_VAD_FILTER,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            results = []
            with tqdm(total=round(info.duration), desc="Transcribing", unit="sec") as pbar:
                last_pos = 0
                for segment in segments:
                    results.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    })
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
