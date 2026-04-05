"""
Timestamp Alignment Module
- Propagates Whisper word-level timestamps to translated segments
- Handles length mismatches between source and target tokens
- Provides phrase-level merging for natural subtitle reading
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TimestampAligner:
    """
    Aligns translated text segments with their original timestamps.

    Strategy:
    1. Use segment-level timestamps directly from Whisper (start/end are reliable).
    2. For word-level alignment of the translated text, distribute time
       proportionally across translated tokens.
    3. Merge short segments that are too brief to read.
    """

    def __init__(self, min_duration: float = 1.0, max_chars_per_sub: int = 80):
        self.min_duration = min_duration
        self.max_chars = max_chars_per_sub

    def align(self, segments: List[Dict]) -> List[Dict]:
        """
        Input segments must have: start, end, original_text, translated_text.
        Returns enriched segments with word-level alignment for translated text.
        """
        aligned = []
        for seg in segments:
            enriched = self._align_segment(seg)
            aligned.append(enriched)

        aligned = self._merge_short(aligned)
        aligned = self._split_long(aligned)
        return aligned

    # ------------------------------------------------------------------
    def _align_segment(self, seg: Dict) -> Dict:
        """Distribute translated words proportionally over the segment duration."""
        start = seg["start"]
        end = seg["end"]
        duration = max(end - start, 0.01)
        translated = seg.get("translated_text", "")
        words = translated.split()
        n = len(words)

        word_timestamps = []
        if n > 0:
            time_per_word = duration / n
            for i, w in enumerate(words):
                word_timestamps.append({
                    "word": w,
                    "start": round(start + i * time_per_word, 3),
                    "end": round(start + (i + 1) * time_per_word, 3),
                })

        return {
            **seg,
            "translated_words": word_timestamps,
        }

    def _merge_short(self, segments: List[Dict]) -> List[Dict]:
        """Merge adjacent segments shorter than min_duration."""
        if not segments:
            return segments
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if (seg["end"] - prev["start"]) < self.min_duration * 2:
                # Merge
                merged[-1] = {
                    **prev,
                    "end": seg["end"],
                    "text": prev.get("text", "") + " " + seg.get("text", ""),
                    "original_text": prev.get("original_text", "") + " " + seg.get("original_text", ""),
                    "translated_text": prev.get("translated_text", "") + " " + seg.get("translated_text", ""),
                    "translated_words": prev.get("translated_words", []) + seg.get("translated_words", []),
                }
            else:
                merged.append(seg)
        return merged

    def _split_long(self, segments: List[Dict]) -> List[Dict]:
        """Split segments whose translated text exceeds max_chars."""
        result = []
        for seg in segments:
            text = seg.get("translated_text", "")
            if len(text) <= self.max_chars:
                result.append(seg)
                continue
            # Split at roughly the midpoint word boundary
            words = text.split()
            mid = len(words) // 2
            half_time = (seg["start"] + seg["end"]) / 2

            seg1 = {**seg,
                    "end": half_time,
                    "translated_text": " ".join(words[:mid]),
                    "translated_words": seg.get("translated_words", [])[:mid]}
            seg2 = {**seg,
                    "start": half_time,
                    "translated_text": " ".join(words[mid:]),
                    "translated_words": seg.get("translated_words", [])[mid:]}
            result.extend([seg1, seg2])
        return result
