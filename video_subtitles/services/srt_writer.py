"""Utilities to write translated segments into an SRT file."""

from pathlib import Path
from typing import Any, Dict, List

from utils.time_format import format_timestamp

MAX_LINE_CHARS = 24
MIN_DURATION_SEC = 0.4


def _wrap_subtitle_text(text: str, max_line_chars: int = MAX_LINE_CHARS) -> str:
    """Wrap a long single-line subtitle into at most two balanced lines."""
    cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(cleaned) <= max_line_chars or "\n" in text.strip():
        return text.strip() if "\n" in text.strip() else cleaned

    # Prefer splitting near the midpoint at a space boundary
    if " " in cleaned:
        mid = len(cleaned) // 2
        best_pos = -1
        best_dist = len(cleaned)
        for idx, ch in enumerate(cleaned):
            if ch == " ":
                dist = abs(idx - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = idx
        if best_pos > 0:
            left = cleaned[:best_pos].strip()
            right = cleaned[best_pos + 1 :].strip()
            if left and right:
                return f"{left}\n{right}"

    return cleaned


def _sanitize_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop empty segments, ensure positive durations, and fix overlaps."""
    valid: List[Dict[str, Any]] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = max(0.0, float(seg.get("start", 0.0)))
        end = max(start + MIN_DURATION_SEC, float(seg.get("end", start + MIN_DURATION_SEC)))
        valid.append({"start": start, "end": end, "text": text})

    valid.sort(key=lambda s: (s["start"], s["end"]))

    for i in range(len(valid) - 1):
        if valid[i]["end"] > valid[i + 1]["start"]:
            # Clamp end just before the next subtitle starts, keeping a tiny positive span
            valid[i]["end"] = max(valid[i]["start"] + 0.1, valid[i + 1]["start"] - 0.01)

    return valid


def write_srt(segments: List[Dict[str, Any]], output_path: Path) -> Path:
    """
    Persist a list of segments (start, end, text) as an SRT file.
    """
    lines = []
    sanitized = _sanitize_segments(segments)

    for index, segment in enumerate(sanitized, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = _wrap_subtitle_text(segment["text"])
        lines.append(f"{index}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line separator

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
