"""
Whisper ASR Module
- Uses faster-whisper (CTranslate2 backend)
- Returns timestamped word-level segments
- Language auto-detection
- Hallucination filtering for music
"""

import logging
import unicodedata
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class WhisperASR:

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._backend = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            logger.info(f"  Loading faster-whisper [{self.model_size}] on {self.device} ({self.compute_type})")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._backend = "faster_whisper"
        except ImportError:
            import whisper
            logger.info(f"  Loading openai-whisper [{self.model_size}]")
            self._model = whisper.load_model(self.model_size)
            self._backend = "openai_whisper"

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        self._load_model()
        if self._backend == "faster_whisper":
            return self._transcribe_faster(audio_path)
        else:
            return self._transcribe_openai(audio_path)

    def _is_hallucinated(self, text: str) -> bool:
        """
        Detect hallucinated segments - Whisper mixing scripts (Korean in French etc.)
        Returns True if segment should be discarded.
        """
        if not text:
            return True
        # Count distinct unicode script blocks in alphabetic chars
        scripts = set()
        for c in text:
            if c.isalpha():
                try:
                    name = unicodedata.name(c, "")
                    script = name.split()[0] if name else "UNKNOWN"
                    scripts.add(script)
                except Exception:
                    pass
        # More than 2 different scripts = hallucination (e.g. LATIN + HANGUL)
        if len(scripts) > 2:
            return True
        return False

    def _transcribe_faster(self, audio_path: str) -> Dict:
        segments_iter, info = self._model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            # FIX: vad_filter OFF - was dropping first 17s of songs
            vad_filter=False,
            # FIX: condition True - reduces cross-script hallucination
            condition_on_previous_text=True,
            # FIX: short prompt only - was being transcribed literally before
            initial_prompt="Transcribe the song lyrics only.",
            # FIX: -1.0 - don't drop low-confidence singing segments
            log_prob_threshold=-1.0,
            # FIX: 0.6 - was 0.8 which skipped actual singing
            no_speech_threshold=0.6,
            temperature=0.0,
            compression_ratio_threshold=2.4,
        )

        segments = []
        for i, seg in enumerate(segments_iter):
            # Skip near-certain non-speech
            if seg.no_speech_prob > 0.85:
                continue
            text = seg.text.strip()
            if not text:
                continue
            # Skip hallucinated mixed-script garbage
            if self._is_hallucinated(text):
                logger.warning(f"  Skipping hallucinated segment: {text[:50]}")
                continue

            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    })
            segments.append({
                "id": i,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": text,
                "words": words,
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
            })

        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "segments": segments,
        }

    def _transcribe_openai(self, audio_path: str) -> Dict:
        import whisper
        result = self._model.transcribe(
            audio_path,
            word_timestamps=True,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
        )
        lang = result.get("language", "unknown")
        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            if seg.get("no_speech_prob", 0) > 0.85:
                continue
            text = seg["text"].strip()
            if not text:
                continue
            if self._is_hallucinated(text):
                continue
            words = []
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"],
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "probability": round(w.get("probability", 1.0), 4),
                })
            segments.append({
                "id": i,
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": text,
                "words": words,
                "avg_logprob": round(seg.get("avg_logprob", 0.0), 4),
                "no_speech_prob": round(seg.get("no_speech_prob", 0.0), 4),
            })
        return {"language": lang, "language_probability": 0.99, "segments": segments}

    @staticmethod
    def list_models() -> List[str]:
        return ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
