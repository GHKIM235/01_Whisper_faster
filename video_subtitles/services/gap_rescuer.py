import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from services.transcriber import WhisperTranscriber


class GapRescuer:
    """
    Analyzes gaps between subtitle segments and attempts to recover missed speech.
    Uses lightweight RMS energy probing (inspired by 'subtitle_gap_rescue_v1') to
    target active audio islands and deduplicates overlaps with existing segments.
    """

    def __init__(
        self,
        transcriber: WhisperTranscriber,
        min_gap_sec: float = 6.0,
        active_rms_threshold: float = 0.015,
        min_island_sec: float = 0.45,
        merge_gap_sec: float = 0.5,
        context_pad_sec: float = 0.5,
    ):
        self.transcriber = transcriber
        self.min_gap_sec = min_gap_sec
        self.active_rms_threshold = active_rms_threshold
        self.min_island_sec = min_island_sec
        self.merge_gap_sec = merge_gap_sec
        self.context_pad_sec = context_pad_sec

    def _find_active_islands(
        self, audio_path: Path, gap_start: float, gap_end: float
    ) -> List[Tuple[float, float]]:
        """Return active sub-intervals inside [gap_start, gap_end] via 100ms RMS probing."""
        try:
            with wave.open(str(audio_path), "rb") as wf:
                if wf.getsampwidth() != 2:
                    return [(gap_start, gap_end)]
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                total_frames = wf.getnframes()

                start_frame = max(0, int(gap_start * sr))
                end_frame = min(total_frames, int(gap_end * sr))
                if end_frame <= start_frame:
                    return []

                wf.setpos(start_frame)
                raw = wf.readframes(end_frame - start_frame)
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if n_channels > 1:
                    samples = samples.reshape(-1, n_channels).mean(axis=1)
        except Exception:
            # Non-PCM or unreadable WAV: fall back to scanning the full gap
            return [(gap_start, gap_end)]

        hop = max(1, int(sr * 0.1))  # 100ms frames
        n_hops = len(samples) // hop
        if n_hops == 0:
            return []

        frames = samples[: n_hops * hop].reshape(n_hops, hop)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Adaptive floor: handle both quiet recordings and noisy backgrounds
        noise_floor = float(np.percentile(rms, 20)) if len(rms) >= 5 else 0.0
        threshold = max(self.active_rms_threshold, noise_floor * 2.2)
        active = rms >= threshold

        islands: List[Tuple[float, float]] = []
        in_island = False
        isl_start = 0.0
        for idx, is_act in enumerate(active):
            t = gap_start + idx * 0.1
            if is_act and not in_island:
                in_island = True
                isl_start = t
            elif not is_act and in_island:
                in_island = False
                islands.append((isl_start, t))
        if in_island:
            islands.append((isl_start, gap_start + n_hops * 0.1))

        if not islands:
            return []

        # Merge nearby islands and filter out very short blips
        merged: List[Tuple[float, float]] = [islands[0]]
        for s, e in islands[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= self.merge_gap_sec:
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))

        padded: List[Tuple[float, float]] = []
        for s, e in merged:
            if e - s < self.min_island_sec:
                continue
            ps = max(gap_start, s - self.context_pad_sec)
            pe = min(gap_end, e + self.context_pad_sec)
            if padded and ps <= padded[-1][1]:
                padded[-1] = (padded[-1][0], max(padded[-1][1], pe))
            else:
                padded.append((ps, pe))

        return padded

    @staticmethod
    def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
        dur = max(1e-6, a_end - a_start)
        return inter / dur

    def _filter_non_overlapping(
        self,
        existing: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        gap_start: float,
        gap_end: float,
    ) -> List[Dict[str, Any]]:
        """Drop rescued segments that overlap existing ones or repeat adjacent text."""
        kept: List[Dict[str, Any]] = []
        for cand in candidates:
            cs = max(gap_start, float(cand["start"]))
            ce = min(gap_end, float(cand["end"]))
            text = str(cand.get("text", "")).strip()
            if not text or ce - cs < 0.25:
                continue

            is_dup = False
            for ex in existing:
                if self._overlap_ratio(cs, ce, ex["start"], ex["end"]) >= 0.35:
                    is_dup = True
                    break
                # Also skip exact text duplicates within 2 seconds of an existing segment
                if abs(cs - ex["end"]) < 2.0 or abs(ex["start"] - ce) < 2.0:
                    ex_text = str(ex.get("text", "")).strip()
                    if ex_text and (text == ex_text or (len(text) >= 4 and text in ex_text)):
                        is_dup = True
                        break
            if not is_dup:
                kept.append({**cand, "start": cs, "end": ce, "text": text, "rescued": True})
        return kept

    def rescue(self, segments: List[Dict[str, Any]], audio_path: Path) -> List[Dict[str, Any]]:
        """
        Finds gaps in segments and re-transcribes active audio islands.
        Returns a combined and sorted list of original and rescued segments.
        """
        if not segments:
            return []

        print(f"\n[GAP_RESCUE] Scanning for gaps > {self.min_gap_sec}s...")

        gaps: List[Tuple[float, float]] = []
        if segments[0]["start"] > self.min_gap_sec:
            gaps.append((0.0, segments[0]["start"]))

        for i in range(len(segments) - 1):
            gap_start = segments[i]["end"]
            gap_end = segments[i + 1]["start"]
            if gap_end - gap_start > self.min_gap_sec:
                gaps.append((gap_start, gap_end))

        if not gaps:
            print("[GAP_RESCUE] No significant gaps found.")
            return segments

        print(f"[GAP_RESCUE] Found {len(gaps)} potential gaps. Probing active audio...")

        rescued_segments: List[Dict[str, Any]] = []
        for gap_start, gap_end in gaps:
            islands = self._find_active_islands(audio_path, gap_start, gap_end)
            if not islands:
                continue
            for isl_start, isl_end in islands:
                new_segments = self.transcriber.transcribe_gap(audio_path, isl_start, isl_end)
                filtered = self._filter_non_overlapping(
                    segments + rescued_segments, new_segments, gap_start, gap_end
                )
                if filtered:
                    print(f"  -> Rescued {len(filtered)} new segments in {isl_start:.1f}s-{isl_end:.1f}s")
                    rescued_segments.extend(filtered)

        if not rescued_segments:
            print("[GAP_RESCUE] No new dialogue recovered.")
            return segments

        combined = segments + rescued_segments
        combined.sort(key=lambda x: x["start"])

        print(f"[GAP_RESCUE] Success: Added {len(rescued_segments)} segments. Total: {len(combined)}")
        return combined
