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
        
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_info_available = True
        except:
            pass

        if config.DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.DEVICE
            
        if config.COMPUTE_TYPE == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = config.COMPUTE_TYPE
            
        print(f"[INIT] Whisper [Device: {self.device} | Precision: {self.compute_type} | Model: {self.model_name}]")
        self.model = self._load_model()

    def _get_gpu_stats(self) -> str:
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

    @staticmethod
    def _is_cjk_language(lang: Optional[str]) -> bool:
        return (lang or "").lower() in {"ja", "zh", "ko", "yue"}

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        """Filter out empty, punctuation-only, or heavily repeated hallucination artifacts."""
        stripped = text.strip()
        if not stripped:
            return True
        if all(ch in ".,!?。、！？…・ー-~〜♪♫()[]「」『』 " for ch in stripped):
            return True
        # Detect short phrase repeated 4+ times consecutively (e.g. "ああああ" / "はい はい はい はい")
        if len(stripped) >= 8:
            for unit_len in (1, 2, 3, 4):
                unit = stripped[:unit_len]
                if unit.strip() and stripped.count(unit) * unit_len >= len(stripped) * 0.85 and len(stripped) // unit_len >= 5:
                    return True
        return False

    def _split_segment_words(self, segment: Any) -> List[Dict[str, Any]]:
        """Split a Whisper segment into natural, readable subtitle chunks."""
        max_segment_duration = 6.5
        max_gap_duration = 1.0
        max_chars = 36
        sentence_end_punct = set("。！？.!?")
        clause_punct = set("、,，")
        is_cjk = self._is_cjk_language(config.SOURCE_LANGUAGE)

        if not getattr(segment, "words", None):
            text = segment.text.strip()
            if self._is_hallucination(text):
                return []
            return [{"start": segment.start, "end": segment.end, "text": text}]

        chunks: List[Dict[str, Any]] = []
        current_tokens: List[str] = []
        sub_start = segment.words[0].start

        for i, word in enumerate(segment.words):
            token = word.word.strip() if is_cjk else word.word
            if token:
                current_tokens.append(token)

            joined = ("".join(current_tokens) if is_cjk else "".join(current_tokens).strip()).strip()
            is_last_word = i == len(segment.words) - 1
            gap_to_next = 0.0 if is_last_word else max(0.0, segment.words[i + 1].start - word.end)
            duration_so_far = word.end - sub_start
            last_char = token[-1] if token else ""

            hit_sentence_end = last_char in sentence_end_punct and (duration_so_far >= 0.9 or len(joined) >= 8)
            hit_clause_limit = last_char in clause_punct and len(joined) >= 22
            hit_hard_limit = gap_to_next > max_gap_duration or duration_so_far > max_segment_duration or len(joined) >= max_chars

            if is_last_word or hit_sentence_end or hit_clause_limit or hit_hard_limit:
                if joined and not self._is_hallucination(joined):
                    chunks.append({
                        "start": sub_start,
                        "end": max(sub_start + 0.2, word.end),
                        "text": joined,
                    })
                if not is_last_word:
                    sub_start = segment.words[i + 1].start
                    current_tokens = []

        return chunks

    def transcribe(self, audio_path: Path, mode: str = "movie") -> List[Dict[str, Any]]:
        print(f"[RUN] Starting Analysis [Mode: {mode.upper()}] : {audio_path.name}")
        is_ja = (config.SOURCE_LANGUAGE or "").lower() == "ja"

        if mode.lower() == "music":
            vad_filter = False
            initial_prompt = "これは歌の歌詞です。正確に書き起こしてください。" if is_ja else None
            repetition_penalty = 1.4
            no_speech_threshold = 0.6
            condition_on_previous_text = False
            hallucination_silence_threshold = None
            vad_params = None
        else:
            # Movie/Dialogue settings: Natural readability and hallucination control
            vad_filter = config.USE_VAD_FILTER
            initial_prompt = "これは映画のセリフです。句読点をつけて自然に書き起こしてください。" if is_ja else None
            repetition_penalty = 1.1
            no_speech_threshold = 0.6
            condition_on_previous_text = False
            hallucination_silence_threshold = 2.0 if vad_filter else None
            vad_params = dict(
                threshold=0.42,              # Slightly higher recall for softer dialogue
                min_silence_duration_ms=500, # Smoother transition
                min_speech_duration_ms=200,
                speech_pad_ms=300,           # Enough padding for natural fade
            )

        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                language=config.SOURCE_LANGUAGE,
                task="transcribe",
                beam_size=5,
                vad_filter=vad_filter,
                vad_parameters=vad_params,
                initial_prompt=initial_prompt,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=3,
                no_speech_threshold=no_speech_threshold,
                log_prob_threshold=-1.0,
                condition_on_previous_text=condition_on_previous_text,
                hallucination_silence_threshold=hallucination_silence_threshold,
                word_timestamps=True,
            )

            results: List[Dict[str, Any]] = []
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

            with tqdm(total=round(info.duration), desc="[ANALYZING]", unit="sec", bar_format=bar_format) as pbar:
                last_pos = 0.0
                for segment in segments:
                    results.extend(self._split_segment_words(segment))
                    delta = max(0, round(segment.end - last_pos))
                    if delta:
                        pbar.update(delta)
                    last_pos = max(last_pos, segment.end)
                    pbar.set_postfix_str(self._get_gpu_stats())
            return results
        except RuntimeError as e:
            if "CUDNN_STATUS_NOT_INITIALIZED" in str(e) and self.device == "cuda":
                print("\n[WARN] cuDNN failure. Falling back to CPU...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = self._load_model()
                return self.transcribe(audio_path, mode=mode)
            raise e

    def transcribe_gap(self, audio_path: Path, start_sec: float, end_sec: float) -> List[Dict[str, Any]]:
        """High-recall transcription for a specific time range to rescue missed dialogue."""
        duration = end_sec - start_sec
        if duration < 0.5:  # Too short to analyze
            return []

        print(f"[RESCUE] Re-analyzing Gap: {start_sec:.2f}s - {end_sec:.2f}s ({duration:.2f}s)")
        is_ja = (config.SOURCE_LANGUAGE or "").lower() == "ja"

        try:
            # High-recall settings for gap rescue
            segments, _ = self.model.transcribe(
                str(audio_path),
                language=config.SOURCE_LANGUAGE,
                clip_timestamps=[float(start_sec), float(end_sec)],
                vad_filter=False,  # Disable VAD to catch faint speech
                initial_prompt="これは聞き取りにくいセリフです。漏らさず書き起こしてください。" if is_ja else None,
                no_speech_threshold=0.35,  # More aggressive
                log_prob_threshold=-1.0,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                condition_on_previous_text=False,
                word_timestamps=True,
            )

            results: List[Dict[str, Any]] = []
            for segment in segments:
                results.extend(self._split_segment_words(segment))
            return results
        except Exception as e:
            print(f"[RESCUE_ERROR] Failed to analyze gap: {e}")
            return []
