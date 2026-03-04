"""Whisper transcription service optimized with hardware control settings."""

import os
import torch
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from faster_whisper import WhisperModel
from tqdm.auto import tqdm
import pynvml

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
        self.gpu_info_available = False
        
        # Initialize NVML for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_info_available = True
        except:
            pass

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
            
        print(f"[INIT] Whisper [Device: {self.device} | Precision: {self.compute_type} | Model: {self.model_name}]")
        self.model = self._load_model()

    def _get_gpu_stats(self) -> str:
        """Returns a string representing current GPU usage."""
        if not self.gpu_info_available or self.device != "cuda":
            return ""
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            vram_gb = mem.used / (1024**3)
            return f" | GPU: {util:2d}% | VRAM: {vram_gb:4.1f}GB"
        except:
            return ""

    def _load_model(self) -> WhisperModel:
        return WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: Path, mode: str = "movie") -> List[Dict[str, Any]]:
        print(f"[RUN] Starting Analysis [Mode: {mode.upper()}] : {audio_path.name}")
        
        # --- Mode-specific params ---
        if mode.lower() == "music":
            vad_filter = False
            initial_prompt = "これは歌의 歌詞입니다. 노래 가사(Lyrics)를 정확히 받아쓰기 하세요."
            repetition_penalty = 1.5
            no_speech_threshold = 0.6
            condition_on_previous_text = False
        else:
            vad_filter = config.USE_VAD_FILTER
            initial_prompt = "これは映画の台詞です. 自然に書き起こしてください。"
            repetition_penalty = 1.1
            no_speech_threshold = 0.5
            condition_on_previous_text = True

        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                language=config.SOURCE_LANGUAGE,
                task="transcribe",
                beam_size=5,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=1000) if vad_filter else None,
                initial_prompt=initial_prompt,
                repetition_penalty=repetition_penalty,
                no_speech_threshold=no_speech_threshold,
                log_prob_threshold=-1.0,
                condition_on_previous_text=condition_on_previous_text
            )
            
            results = []
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            
            with tqdm(total=round(info.duration), desc="[ANALYZING]", unit="sec", bar_format=bar_format) as pbar:
                last_pos = 0
                for segment in segments:
                    results.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    })
                    pbar.update(round(segment.end - last_pos))
                    last_pos = segment.end
                    # Update real-time GPU stats
                    pbar.set_postfix_str(self._get_gpu_stats())
            return results
        except RuntimeError as e:
            if "CUDNN_STATUS_NOT_INITIALIZED" in str(e) and self.device == "cuda":
                print("\n[WARN] cuDNN failure. Falling back to CPU...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = self._load_model()
                return self.transcribe(audio_path)
            raise e
