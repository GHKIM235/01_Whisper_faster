"""Translation layer using multiple engines (Google/DeepL) with multi-key support."""

import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from deep_translator import GoogleTranslator
try:
    import deepl
except ImportError:
    deepl = None

from tqdm.auto import tqdm

# Import central config
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    import config
except ImportError:
    class ConfigMock:
        SOURCE_LANGUAGE = "ja"
        TARGET_LANGUAGE = "ko"
        TRANSLATION_BATCH_SIZE = 50
        MAX_CHARS_PER_BATCH = 4000
        TRANSLATION_ENGINE = "google"
        DEEPL_API_KEY = ""
        WORK_DIR = "work"
    config = ConfigMock()

class JapaneseToKoreanTranslator:
    """Wraps deep-translator and DeepL official SDK with Multi-Key rotation."""

    def __init__(self, source: str = None, target: str = None):
        self.source = (source or config.SOURCE_LANGUAGE).upper()
        self.target = (target or config.TARGET_LANGUAGE).upper()
        self.engine = config.TRANSLATION_ENGINE.lower()
        self.progress_file = Path(config.WORK_DIR) / "translation_progress.json"
        
        # Multi-Key Support
        all_keys = [k.strip() for k in config.DEEPL_API_KEY.split(",") if k.strip()]
        self.key_pool = all_keys
        self.current_key_index = 0
        
        self.deepl_client: Optional[deepl.Translator] = None
        self.google_client = None
        
        self._init_engine()

    def _init_engine(self):
        """Initialize or rotate the translation engine."""
        if self.engine == "deepl" and deepl and self.key_pool:
            try:
                current_key = self.key_pool[self.current_key_index]
                self.deepl_client = deepl.Translator(current_key)
                print(f"[INIT] DeepL Engine (Key {self.current_key_index + 1}/{len(self.key_pool)})")
            except Exception as e:
                print(f"[WARN] Failed to init DeepL Key {self.current_key_index + 1}: {e}")
                self._rotate_key()
        
        if self.engine == "google" or (self.engine == "deepl" and not self.deepl_client):
            self.google_client = GoogleTranslator(source=self.source.lower(), target=self.target.lower())
            print(f"[INIT] Google Translator (Backup)")

    def _rotate_key(self) -> bool:
        """Switch to the next DeepL key if available."""
        self.current_key_index += 1
        if self.current_key_index < len(self.key_pool):
            print(f"🔄 Rotating DeepL API Key... (Switching to Key {self.current_key_index + 1})")
            self._init_engine()
            return True
        else:
            print("🚨 All DeepL API Keys exhausted. Falling back to Google Translate.")
            self.engine = "google"
            self._init_engine()
            return False

    def get_usage(self) -> str:
        """Fetch current DeepL API usage (characters)."""
        if self.engine == "deepl" and self.deepl_client:
            try:
                usage = self.deepl_client.get_usage()
                if usage.character:
                    return f"{usage.character.count:,} / {usage.character.limit:,} characters (Key {self.current_key_index + 1})"
            except: pass
        return "N/A (Google Translate)"

    def _translate_batch(self, texts: List[str]) -> List[str]:
        if not texts: return []
        
        if self.engine == "deepl" and self.deepl_client:
            try:
                result = self.deepl_client.translate_text(
                    texts, 
                    source_lang=self.source, 
                    target_lang=self.target
                )
                if isinstance(result, list):
                    return [r.text for r in result]
                return [result.text] if hasattr(result, 'text') else [str(result)]
            
            except deepl.QuotaExceededException:
                print(f"\n⚠️  Key {self.current_key_index + 1} Quota Exceeded!")
                if self._rotate_key():
                    return self._translate_batch(texts) # Retry with new key
                else:
                    return self._translate_batch(texts) # Will fall back to Google in retry
            except Exception as e:
                print(f"\n[ERROR] DeepL API Error: {e}")
                if self._rotate_key():
                    return self._translate_batch(texts)
                return []
        else:
            # Google
            combined_text = "\n".join(texts)
            translated_batch = self.google_client.translate(combined_text)
            return [line.strip() for line in translated_batch.splitlines()]

    def translate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments: return []

        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        last_completed, completed_map = self._load_progress()
        total_segments = len(segments)

        translated_map: Dict[int, Dict[str, Any]] = {}
        for index, text in completed_map.items():
            if 0 <= index < total_segments:
                translated_map[index] = {**segments[index], "text": text}

        start_index = min(last_completed + 1, total_segments)
        if start_index >= total_segments:
            return [translated_map[i] for i in range(total_segments)]

        progress_bar = tqdm(
            total=total_segments,
            desc=f"Translating ({self.engine})",
            unit="seg",
            initial=start_index,
        )

        current_idx = start_index
        while current_idx < total_segments:
            batch_indices = []
            batch_texts = []
            current_chars = 0
            
            for i in range(current_idx, total_segments):
                text = segments[i]["text"]
                if current_chars + len(text) + 1 > config.MAX_CHARS_PER_BATCH:
                    if not batch_indices:
                        batch_indices.append(i)
                        batch_texts.append(text)
                        current_idx += 1
                    break
                
                batch_indices.append(i)
                batch_texts.append(text)
                current_chars += len(text) + 1
                current_idx += 1
                if len(batch_indices) >= config.TRANSLATION_BATCH_SIZE:
                    break

            if not batch_indices: break

            try:
                translated_lines = self._translate_batch(batch_texts)
                for idx, translated_line in zip(batch_indices, translated_lines):
                    translated_map[idx] = {**segments[idx], "text": translated_line}
                    completed_map[idx] = translated_line
                    last_completed = idx
            except Exception as e:
                print(f"\n[ERROR] Translation flow failed: {e}")
                break

            progress_bar.update(len(batch_indices))
            self._save_progress(last_completed, completed_map)

        progress_bar.close()
        if len(translated_map) == total_segments:
            self._clear_progress()
            
        return [translated_map.get(i, {**segments[i], "text": "[Error]"}) for i in range(total_segments)]

    def _load_progress(self) -> Tuple[int, Dict[int, str]]:
        if not self.progress_file.exists(): return -1, {}
        try:
            data = json.loads(self.progress_file.read_text(encoding="utf-8"))
            return int(data.get("last_completed_index", -1)), {int(idx): text for idx, text in data.get("translations", {}).items()}
        except: return -1, {}

    def _save_progress(self, last_index: int, translations: Dict[int, str]) -> None:
        payload = {"last_completed_index": last_index, "translations": {str(idx): text for idx, text in translations.items()}}
        self.progress_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _clear_progress(self) -> None:
        if self.progress_file.exists(): self.progress_file.unlink()
