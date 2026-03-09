"""Vocal separation service using Demucs to remove background noise."""

import os
import subprocess
from pathlib import Path
import sys

# Import central config
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    import config
except ImportError:
    class ConfigMock:
        USE_VOCAL_SEPARATION = False
        DEVICE = "auto"
    config = ConfigMock()

def separate_vocals(audio_path: Path, output_dir: Path) -> Path:
    """
    Separate vocals from the audio file using Demucs.
    Returns the path to the separated vocal file.
    """
    if not config.USE_VOCAL_SEPARATION:
        return audio_path

    # Output path for demucs is usually output_dir/htdemucs/filename/vocals.wav
    model_name = "htdemucs"
    vocal_output_dir = output_dir / model_name / audio_path.stem
    vocal_path = vocal_output_dir / "vocals.wav"
    
    # Final target path to keep it organized with original audio
    target_vocal_path = output_dir / f"{audio_path.stem}_vocals.wav"

    if target_vocal_path.exists():
        print(f"Vocal file already exists: {target_vocal_path.name}. Skipping separation.")
        return target_vocal_path

    print(f"[PRE-PROCESS] Removing background noise (Vocal Separation)...")
    print(f"This may take a while for the first time (downloading model)...")

    # Determine device
    device = "cuda" if config.DEVICE == "auto" or config.DEVICE == "cuda" else "cpu"
    
    command = [
        sys.executable, "-m", "demucs.separate",
        "-n", model_name,
        "--two-stems", "vocals",  # Only extract vocals and 'other'
        "-o", str(output_dir),
        str(audio_path)
    ]
    
    if device == "cuda":
        command.extend(["-d", "cuda"])
    else:
        command.extend(["-d", "cpu"])

    try:
        # Run demucs
        subprocess.run(command, check=True)
        
        # Demucs creates a nested structure, let's move the vocals.wav to a cleaner location
        if vocal_path.exists():
            if target_vocal_path.exists():
                os.remove(target_vocal_path)
            os.rename(vocal_path, target_vocal_path)
            
            # Optional: Clean up demucs specific folder
            # import shutil
            # shutil.rmtree(output_dir / model_name)
            
            print(f"[SUCCESS] Vocal separation complete: {target_vocal_path.name}")
            return target_vocal_path
        else:
            print(f"[WARN] Demucs finished but vocals.wav not found at {vocal_path}")
            return audio_path

    except Exception as e:
        print(f"[ERROR] Vocal separation failed: {e}")
        print("Continuing with original audio...")
        return audio_path
