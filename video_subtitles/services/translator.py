"""Translation layer that converts Japanese text to Korean text."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from deep_translator import GoogleTranslator
from tqdm.auto import tqdm


class JapaneseToKoreanTranslator:
    """Wraps deep-translator so we can swap implementations if needed."""

    BATCH_SIZE = 10
    PROGRESS_FILE = Path("progress.json")

    def __init__(self, source: str = "ja", target: str = "ko"):
        self.client = GoogleTranslator(source=source, target=target)

    def translate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Translate the text portion of each segment while preserving timings.
        """
        if not segments:
            return []

        last_completed, completed_map = self._load_progress()
        total_segments = len(segments)

        translated_map: Dict[int, Dict[str, Any]] = {}

        # Prefill already translated segments.
        for index, text in completed_map.items():
            if 0 <= index < total_segments:
                translated_map[index] = {**segments[index], "text": text}

        start_index = min(last_completed + 1, total_segments)
        progress_bar = tqdm(
            total=total_segments,
            desc="Translating segments",
            unit="segment",
            initial=start_index,
        )

        for batch_start in range(start_index, total_segments, self.BATCH_SIZE):
            indices = list(range(batch_start, min(batch_start + self.BATCH_SIZE, total_segments)))
            batch = [segments[i] for i in indices]
            batch_text = "\n".join(item["text"] for item in batch)

            translated_batch = self.client.translate(batch_text)
            translated_lines = [line.strip() for line in translated_batch.splitlines()]

            for idx, translated_line in zip(indices, translated_lines):
                translated_map[idx] = {
                    **segments[idx],
                    "text": translated_line,
                }
                completed_map[idx] = translated_line
                last_completed = idx

            progress_bar.update(len(indices))
            progress_bar.set_postfix_str(f"{min(indices[-1] + 1, total_segments)}/{total_segments}")
            self._save_progress(last_completed, completed_map)

        progress_bar.close()

        # Clean up progress if finished and return the completed list.
        if len(translated_map) == total_segments and total_segments > 0:
            self._clear_progress()
        return [translated_map[i] if i in translated_map else {**segments[i], "text": ""} for i in range(total_segments)]

    def _load_progress(self) -> Tuple[int, Dict[int, str]]:
        """
        Load progress.json to determine the last completed segment, if any.
        """
        if not self.PROGRESS_FILE.exists():
            return -1, {}

        try:
            data = json.loads(self.PROGRESS_FILE.read_text(encoding="utf-8"))
            last_index = int(data.get("last_completed_index", -1))
            translations = {
                int(idx): text for idx, text in data.get("translations", {}).items()
            }
            return last_index, translations
        except (ValueError, json.JSONDecodeError):
            return -1, {}

    def _save_progress(self, last_index: int, translations: Dict[int, str]) -> None:
        payload = {
            "last_completed_index": last_index,
            "translations": {str(idx): text for idx, text in translations.items()},
        }
        self.PROGRESS_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _clear_progress(self) -> None:
        if self.PROGRESS_FILE.exists():
            self.PROGRESS_FILE.unlink()
