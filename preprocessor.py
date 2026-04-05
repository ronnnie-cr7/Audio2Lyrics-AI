"""
Audio Preprocessing Module
- Resampling, mono conversion, normalisation
- Noise reduction (noisereduce)
- Optional vocal isolation (demucs)
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Prepares raw audio for ASR."""

    SUPPORTED = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}

    def __init__(self, target_sr: int = 16000, denoise: bool = True, remove_vocals: bool = False):
        self.target_sr = target_sr
        self.denoise = denoise
        self.remove_vocals = remove_vocals

        if remove_vocals:
            try:
                import demucs  # noqa
                self._demucs_available = True
            except ImportError:
                logger.warning("demucs not installed – vocal separation disabled.")
                self._demucs_available = False
        else:
            self._demucs_available = False

    # ------------------------------------------------------------------
    def process(self, audio_path: str) -> Tuple[str, float]:
        """
        Full preprocessing chain.
        Returns (processed_wav_path, duration_seconds).
        """
        ext = Path(audio_path).suffix.lower()
        if ext not in self.SUPPORTED:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.SUPPORTED}")

        logger.info(f"  Loading {audio_path}")
        y, sr = self._load(audio_path)
        duration = len(y) / sr

        logger.info(f"  Duration: {duration:.1f}s | SR: {sr} → {self.target_sr}")
        y = self._to_mono(y)
        y = self._resample(y, sr)
        y = self._normalise(y)

        if self.denoise:
            y = self._denoise(y)

        if self.remove_vocals and self._demucs_available:
            y = self._isolate_vocals(audio_path, y)

        out_path = self._save_temp(y)
        return out_path, duration

    # ------------------------------------------------------------------
    def _load(self, path: str) -> Tuple[np.ndarray, int]:
        ext = Path(path).suffix.lower()
        if ext != ".wav":
            seg = AudioSegment.from_file(path)
            tmp = tempfile.mktemp(suffix=".wav")
            seg.export(tmp, format="wav")
            path = tmp
        y, sr = librosa.load(path, sr=None, mono=False)
        return y, sr

    def _to_mono(self, y: np.ndarray) -> np.ndarray:
        if y.ndim > 1:
            return librosa.to_mono(y)
        return y

    def _resample(self, y: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.target_sr:
            return librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
        return y

    def _normalise(self, y: np.ndarray) -> np.ndarray:
        """Peak normalisation to -1 dBFS."""
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.9
        return y

    def _denoise(self, y: np.ndarray) -> np.ndarray:
        """Spectral noise reduction using noisereduce."""
        try:
            import noisereduce as nr
            # Use first 0.5 s as noise profile (silence / intro)
            noise_sample = y[: self.target_sr // 2]
            return nr.reduce_noise(y=y, sr=self.target_sr, y_noise=noise_sample, prop_decrease=0.8)
        except ImportError:
            logger.warning("noisereduce not installed – skipping denoising.")
            return y

    def _isolate_vocals(self, original_path: str, fallback: np.ndarray) -> np.ndarray:
        """Runs demucs htdemucs model to extract vocals stem."""
        try:
            import subprocess, glob
            out_dir = tempfile.mkdtemp()
            subprocess.run(
                ["python", "-m", "demucs", "--two-stems=vocals", "-o", out_dir, original_path],
                check=True, capture_output=True,
            )
            vocal_files = glob.glob(os.path.join(out_dir, "**", "vocals.wav"), recursive=True)
            if vocal_files:
                y, _ = librosa.load(vocal_files[0], sr=self.target_sr, mono=True)
                logger.info("  Vocal stem extracted via demucs.")
                return y
        except Exception as e:
            logger.warning(f"  Vocal isolation failed: {e} – using full mix.")
        return fallback

    def _save_temp(self, y: np.ndarray) -> str:
        tmp = tempfile.mktemp(suffix="_processed.wav")
        sf.write(tmp, y, self.target_sr)
        return tmp

    # ------------------------------------------------------------------
    @staticmethod
    def get_audio_info(path: str) -> dict:
        """Returns metadata dict without full load."""
        y, sr = librosa.load(path, sr=None, mono=True, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return {
            "sample_rate": sr,
            "duration": librosa.get_duration(path=path),
            "estimated_tempo_bpm": float(tempo),
            "channels": 1,
        }
