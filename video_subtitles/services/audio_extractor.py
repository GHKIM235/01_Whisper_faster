"""Extract an audio track from a video file using ffmpeg."""

from pathlib import Path
import subprocess


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """
    Extract mono 16kHz WAV audio from the video.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{video_path.stem}_audio.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]

    subprocess.run(command, check=True)
    return audio_path

