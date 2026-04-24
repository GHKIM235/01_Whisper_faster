import sys
from pathlib import Path
from typing import List, Dict, Any
import time

# Import transcriber for type hinting and internal use
from services.transcriber import WhisperTranscriber

class GapRescuer:
    """
    Analyzes gaps between subtitle segments and attempts to recover missed speech.
    Inspired by the 'subtitle_gap_rescue_v1' schema.
    """
    def __init__(self, transcriber: WhisperTranscriber, min_gap_sec: float = 6.0):
        self.transcriber = transcriber
        self.min_gap_sec = min_gap_sec

    def rescue(self, segments: List[Dict[str, Any]], audio_path: Path) -> List[Dict[str, Any]]:
        """
        Finds gaps in segments and re-transcribes them.
        Returns a combined and sorted list of original and rescued segments.
        """
        if not segments:
            return []

        print(f"\n[GAP_RESCUE] Scanning for gaps > {self.min_gap_sec}s...")
        
        gaps = []
        # Check gap before first segment
        if segments[0]['start'] > self.min_gap_sec:
            gaps.append((0.0, segments[0]['start']))

        # Check gaps between segments
        for i in range(len(segments) - 1):
            gap_start = segments[i]['end']
            gap_end = segments[i+1]['start']
            if gap_end - gap_start > self.min_gap_sec:
                gaps.append((gap_start, gap_end))

        # We don't check trailing gap here as it's often end credits/silence
        
        if not gaps:
            print("[GAP_RESCUE] No significant gaps found.")
            return segments

        print(f"[GAP_RESCUE] Found {len(gaps)} potential gaps. Starting re-analysis...")
        
        rescued_segments = []
        for start, end in gaps:
            # We add a small buffer/padding to avoid clipping words
            # but WhisperTranscriber.transcribe_gap handles the specific range.
            new_segments = self.transcriber.transcribe_gap(audio_path, start, end)
            if new_segments:
                print(f"  -> Rescued {len(new_segments)} new segments in gap {start:.1f}s-{end:.1f}s")
                for s in new_segments:
                    s['rescued'] = True # Mark as rescued for logging/tracking
                rescued_segments.extend(new_segments)

        if not rescued_segments:
            print("[GAP_RESCUE] No new dialogue recovered.")
            return segments

        # Merge and sort
        combined = segments + rescued_segments
        combined.sort(key=lambda x: x['start'])
        
        print(f"[GAP_RESCUE] Success: Added {len(rescued_segments)} segments. Total: {len(combined)}")
        return combined
