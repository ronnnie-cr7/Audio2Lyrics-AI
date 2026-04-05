"""
Whisper ASR Module
- Uses faster-whisper (CTranslate2 backend) for speed & efficiency
- Returns timestamped word-level segments
- Language auto-detection
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class WhisperASR:
    """
    Wrapper around faster-whisper for music/speech transcription.
    Falls back to openai-whisper if faster-whisper not available.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            logger.info(f"  Loading faster-whisper [{self.model_size}] on {self.device} ({self.compute_type})")
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            self._backend = "faster_whisper"
        except ImportError:
            import whisper
            logger.info(f"  Loading openai-whisper [{self.model_size}]")
            self._model = whisper.load_model(self.model_size)
            self._backend = "openai_whisper"

    # ------------------------------------------------------------------
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Returns:
            {
                "language": str,
                "language_probability": float,
                "segments": [
                    {
                        "id": int,
                        "start": float,
                        "end": float,
                        "text": str,
                        "words": [{"word": str, "start": float, "end": float, "probability": float}],
                        "avg_logprob": float,
                        "no_speech_prob": float,
                    }
                ]
            }
        """
        self._load_model()

        if self._backend == "faster_whisper":
            return self._transcribe_faster(audio_path)
        else:
            return self._transcribe_openai(audio_path)

    # ------------------------------------------------------------------
    def _transcribe_faster(self, audio_path: str) -> Dict:
        segments_iter, info = self._model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,                        # skip silence / non-speech
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=True,
            initial_prompt=(
                "Song lyrics. Musical performance. "
                "Include slang, informal contractions, and rap lyrics accurately."
            ),
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        segments = []
        for i, seg in enumerate(segments_iter):
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
                "text": seg.text.strip(),
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
            initial_prompt="Song lyrics. Musical performance.",
        )
        lang = result.get("language", "unknown")
        segments = []
        for i, seg in enumerate(result.get("segments", [])):
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
                "text": seg["text"].strip(),
                "words": words,
                "avg_logprob": round(seg.get("avg_logprob", 0.0), 4),
                "no_speech_prob": round(seg.get("no_speech_prob", 0.0), 4),
            })
        return {"language": lang, "language_probability": 0.99, "segments": segments}

    # ------------------------------------------------------------------
    @staticmethod
    def list_models() -> List[str]:
        return ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
