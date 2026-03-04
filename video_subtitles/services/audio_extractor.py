"""Extract an audio track from a video file using ffmpeg."""

import os
from pathlib import Path
import subprocess


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """
    Extract mono 16kHz WAV audio from the video.
    Returns the path to the extracted audio file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{video_path.stem}_audio.wav"

    # Optimization: Skip if the audio file already exists (useful if re-running)
    if audio_path.exists():
        print(f"Audio file already exists: {audio_path.name}. Skipping extraction.")
        return audio_path

    print(f"Extracting audio from: {video_path.name}")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",          # No video
        "-ac",
        "1",           # Mono
        "-ar",
        "16000",       # 16kHz
        str(audio_path),
    ]

    try:
        # Run quietly to keep the terminal clean
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed: {e}")
        raise RuntimeError(f"Could not extract audio from {video_path}")

    return audio_path
