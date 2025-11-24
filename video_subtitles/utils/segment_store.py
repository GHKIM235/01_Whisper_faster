"""Helpers for persisting transcription segments for later translation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def save_segments(
    segments: List[Dict[str, Any]],
    path: Path,
    *,
    source_video: Optional[Path] = None,
) -> Path:
    """
    Persist the raw segments list so translation can be executed later.
    """
    payload: Dict[str, Any] = {
        "segments": segments,
        "meta": {},
    }
    if source_video is not None:
        payload["meta"]["source_video"] = str(source_video)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    return path


def load_segments(path: Path) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    """
    Load a saved segments file and return both the data and original video path.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    meta = data.get("meta", {})
    source_video = meta.get("source_video")
    return segments, Path(source_video) if source_video else None
