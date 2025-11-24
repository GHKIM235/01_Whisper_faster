"""Utilities to write translated segments into an SRT file."""

from pathlib import Path
from typing import Any, Dict, List

from utils.time_format import format_timestamp


def write_srt(segments: List[Dict[str, Any]], output_path: Path) -> Path:
    """
    Persist a list of segments (start, end, text) as an SRT file.
    """
    lines = []

    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"]
        lines.append(f"{index}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line separator

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
