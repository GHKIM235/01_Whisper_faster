"""Utility helpers to format timestamps for SRT output."""


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds (accepts floats) into the SRT HH:MM:SS,mmm format.
    """
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
