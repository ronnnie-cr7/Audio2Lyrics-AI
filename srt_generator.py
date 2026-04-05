"""
SRT Subtitle File Generator
Converts aligned segments to standard SRT format.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def _format_ts(seconds: float) -> str:
    """Convert float seconds → SRT timestamp HH:MM:SS,mmm"""
    ms = int(round((seconds % 1) * 1000))
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


class SRTGenerator:
    def __init__(self, max_chars: int = 42):
        self.max_chars = max_chars

    def generate(self, segments: List[Dict], output_path: str) -> None:
        lines = []
        counter = 1
        for seg in segments:
            text = seg.get("translated_text", "").strip()
            if not text:
                continue
            start_ts = _format_ts(seg["start"])
            end_ts = _format_ts(seg["end"])
            # Wrap long lines
            wrapped = self._wrap(text)
            lines.append(str(counter))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(wrapped)
            lines.append("")
            counter += 1

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"  SRT written: {output_path} ({counter - 1} entries)")

    def _wrap(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        words = text.split()
        lines, current = [], []
        for w in words:
            if sum(len(x) for x in current) + len(current) + len(w) > self.max_chars:
                lines.append(" ".join(current))
                current = [w]
            else:
                current.append(w)
        if current:
            lines.append(" ".join(current))
        return "\n".join(lines)

    def generate_bilingual(self, segments: List[Dict], output_path: str) -> None:
        """Generates SRT with original + translated stacked."""
        lines = []
        counter = 1
        for seg in segments:
            orig = seg.get("original_text", seg.get("text", "")).strip()
            trans = seg.get("translated_text", "").strip()
            if not trans:
                continue
            start_ts = _format_ts(seg["start"])
            end_ts = _format_ts(seg["end"])
            lines.append(str(counter))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(f"<i>{self._wrap(orig)}</i>")
            lines.append(self._wrap(trans))
            lines.append("")
            counter += 1

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
