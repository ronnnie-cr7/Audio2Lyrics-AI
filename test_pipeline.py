"""
Unit & Integration Tests for Song Translation Pipeline
Run: pytest tests/ -v
"""

import sys
import os
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_silent_wav(duration: float = 3.0, sr: int = 16000) -> str:
    """Creates a silent WAV file for testing without real audio."""
    samples = np.zeros(int(duration * sr), dtype=np.float32)
    # Add tiny noise so it's not completely silent
    samples += np.random.randn(len(samples)) * 0.001
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, samples, sr)
    return tmp


# ---------------------------------------------------------------------------
# Audio Preprocessor Tests
# ---------------------------------------------------------------------------
class TestAudioPreprocessor:
    def test_load_wav(self):
        from audio.preprocessor import AudioPreprocessor
        path = make_silent_wav()
        pp = AudioPreprocessor(denoise=False, remove_vocals=False)
        out_path, duration = pp.process(path)
        assert os.path.exists(out_path)
        assert 2.5 < duration < 3.5
        os.unlink(path)

    def test_normalise(self):
        from audio.preprocessor import AudioPreprocessor
        pp = AudioPreprocessor(denoise=False)
        y = np.array([0.5, -1.0, 0.8, -0.3], dtype=np.float32)
        norm = pp._normalise(y)
        assert np.max(np.abs(norm)) <= 1.0

    def test_to_mono_stereo(self):
        from audio.preprocessor import AudioPreprocessor
        pp = AudioPreprocessor(denoise=False)
        stereo = np.random.randn(2, 1000).astype(np.float32)
        mono = pp._to_mono(stereo)
        assert mono.ndim == 1

    def test_unsupported_format(self):
        from audio.preprocessor import AudioPreprocessor
        pp = AudioPreprocessor(denoise=False)
        with pytest.raises(ValueError, match="Unsupported format"):
            pp.process("song.xyz")


# ---------------------------------------------------------------------------
# Slang Normalizer Tests
# ---------------------------------------------------------------------------
class TestSlangNormalizer:
    def test_known_slang(self):
        from translation.translator import SlangNormalizer
        sn = SlangNormalizer()
        result = sn.normalize("ella es la bichota")
        assert "boss woman" in result.lower()

    def test_no_change_for_standard(self):
        from translation.translator import SlangNormalizer
        sn = SlangNormalizer()
        text = "hello world"
        assert sn.normalize(text) == text

    def test_contraction(self):
        from translation.translator import SlangNormalizer
        sn = SlangNormalizer()
        result = sn.normalize("vamos pa' la playa")
        assert "para" in result or "for" in result


# ---------------------------------------------------------------------------
# SRT Generator Tests
# ---------------------------------------------------------------------------
class TestSRTGenerator:
    def _sample_segments(self):
        return [
            {"start": 0.0, "end": 3.5, "translated_text": "Hello world", "original_text": "Hola mundo"},
            {"start": 3.5, "end": 7.0, "translated_text": "How are you?", "original_text": "¿Cómo estás?"},
        ]

    def test_srt_file_created(self):
        from utils.srt_generator import SRTGenerator
        gen = SRTGenerator()
        segs = self._sample_segments()
        tmp = tempfile.mktemp(suffix=".srt")
        gen.generate(segs, tmp)
        assert os.path.exists(tmp)
        content = open(tmp).read()
        assert "Hello world" in content
        assert "00:00:00,000 --> 00:00:03,500" in content

    def test_timestamp_format(self):
        from utils.srt_generator import _format_ts
        assert _format_ts(0.0) == "00:00:00,000"
        assert _format_ts(3661.5) == "01:01:01,500"

    def test_line_wrapping(self):
        from utils.srt_generator import SRTGenerator
        gen = SRTGenerator(max_chars=20)
        long_text = "This is a very long subtitle line that needs wrapping"
        wrapped = gen._wrap(long_text)
        for line in wrapped.split("\n"):
            assert len(line) <= 20 + 5  # allow some slack at word boundary

    def test_bilingual_srt(self):
        from utils.srt_generator import SRTGenerator
        gen = SRTGenerator()
        segs = self._sample_segments()
        tmp = tempfile.mktemp(suffix=".srt")
        gen.generate_bilingual(segs, tmp)
        content = open(tmp).read()
        assert "<i>" in content  # original in italic


# ---------------------------------------------------------------------------
# Aligner Tests
# ---------------------------------------------------------------------------
class TestAligner:
    def _sample_segments(self):
        return [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Hola mundo", "original_text": "Hola mundo",
             "translated_text": "Hello world", "words": []},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "¿Cómo estás?", "original_text": "¿Cómo estás?",
             "translated_text": "How are you?", "words": []},
        ]

    def test_word_timestamps_generated(self):
        from alignment.aligner import TimestampAligner
        al = TimestampAligner()
        segs = al.align(self._sample_segments())
        assert all("translated_words" in s for s in segs)

    def test_no_overlaps(self):
        from alignment.aligner import TimestampAligner
        al = TimestampAligner()
        segs = al.align(self._sample_segments())
        for i in range(1, len(segs)):
            assert segs[i]["start"] >= segs[i - 1]["end"] - 0.001


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_wer_perfect(self):
        from utils.metrics import EvaluationMetrics
        m = EvaluationMetrics()
        assert m.wer("hello world", "hello world") == 0.0

    def test_wer_all_wrong(self):
        from utils.metrics import EvaluationMetrics
        m = EvaluationMetrics()
        assert m.wer("hello world", "foo bar") > 0.0

    def test_bleu_nonzero(self):
        from utils.metrics import EvaluationMetrics
        m = EvaluationMetrics()
        score = m.bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0

    def test_timing_consistency_ok(self):
        from utils.metrics import EvaluationMetrics
        m = EvaluationMetrics()
        segs = [
            {"start": 0.0, "end": 2.0},
            {"start": 2.5, "end": 5.0},
        ]
        result = m.timing_consistency(segs)
        assert result["ok"] is True

    def test_timing_consistency_overlap(self):
        from utils.metrics import EvaluationMetrics
        m = EvaluationMetrics()
        segs = [
            {"start": 0.0, "end": 3.0},
            {"start": 2.0, "end": 5.0},  # overlaps
        ]
        result = m.timing_consistency(segs)
        assert result["ok"] is False
