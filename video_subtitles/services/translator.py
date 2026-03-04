"""Translation layer using central config."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from deep_translator import GoogleTranslator
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
    config = ConfigMock()

class JapaneseToKoreanTranslator:
    """Wraps deep-translator with config-driven settings."""

    def __init__(self, source: str = None, target: str = None):
        source = source or config.SOURCE_LANGUAGE
        target = target or config.TARGET_LANGUAGE
        self.client = GoogleTranslator(source=source, target=target)
        self.progress_file = Path(config.WORK_DIR) / "translation_progress.json"

    def translate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []

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
            desc=f"Translating ({config.SOURCE_LANGUAGE} -> {config.TARGET_LANGUAGE})",
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

            combined_text = "\n".join(batch_texts)
            try:
                translated_batch = self.client.translate(combined_text)
                translated_lines = [line.strip() for line in translated_batch.splitlines()]
                
                for idx, translated_line in zip(batch_indices, translated_lines):
                    translated_map[idx] = {**segments[idx], "text": translated_line}
                    completed_map[idx] = translated_line
                    last_completed = idx
            except Exception as e:
                print(f"\n[ERROR] Translation failed: {e}")
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
