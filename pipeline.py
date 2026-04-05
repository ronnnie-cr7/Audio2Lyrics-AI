"""
Automatic Song Translation Pipeline
Audio → Preprocessing → ASR → Language Detection → Translation → Alignment → SRT
"""

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from preprocessor import AudioPreprocessor
from whisper_asr import WhisperASR
from translator import LyricsTranslator
from aligner import TimestampAligner
from srt_generator import SRTGenerator
from metrics import EvaluationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # Audio
    target_sr: int = 16000
    remove_vocals: bool = False          # Set True if demucs installed
    denoise: bool = True

    # ASR
    whisper_model: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"      # float16 | int8

    # Translation
    translation_backend: str = "helsinki" # helsinki | deepl | openai | google
    target_language: str = "en"
    handle_slang: bool = True

    # Output
    output_dir: str = "outputs"
    generate_srt: bool = True
    srt_max_chars: int = 42


@dataclass
class PipelineResult:
    audio_path: str
    duration_seconds: float
    detected_language: str
    detected_language_confidence: float
    segments: list = field(default_factory=list)          # raw whisper segments
    translated_segments: list = field(default_factory=list)
    srt_path: Optional[str] = None
    processing_time: float = 0.0
    metrics: dict = field(default_factory=dict)


class SongTranslationPipeline:
    """End-to-end pipeline for automatic song translation."""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info("Initialising pipeline components...")
        self.preprocessor = AudioPreprocessor(
            target_sr=self.config.target_sr,
            denoise=self.config.denoise,
            remove_vocals=self.config.remove_vocals,
        )
        self.asr = WhisperASR(
            model_size=self.config.whisper_model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        self.translator = LyricsTranslator(
            backend=self.config.translation_backend,
            target_lang=self.config.target_language,
            handle_slang=self.config.handle_slang,
        )
        self.aligner = TimestampAligner()
        self.srt_gen = SRTGenerator(max_chars=self.config.srt_max_chars)
        self.metrics = EvaluationMetrics()
        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    def run(self, audio_path: str, reference_lyrics: str = None) -> PipelineResult:
        t0 = time.time()
        audio_path = str(Path(audio_path).resolve())
        stem = Path(audio_path).stem

        logger.info(f"=== Processing: {audio_path} ===")

        # 1. Audio preprocessing
        logger.info("[1/5] Preprocessing audio...")
        processed_path, duration = self.preprocessor.process(audio_path)

        # 2. Speech recognition + language detection
        logger.info("[2/5] Running ASR (Whisper)...")
        asr_result = self.asr.transcribe(processed_path)
        segments = asr_result["segments"]
        detected_lang = asr_result["language"]
        lang_prob = asr_result["language_probability"]
        logger.info(f"       Detected language: {detected_lang} (conf={lang_prob:.2f})")

        # 3. Translation
        logger.info("[3/5] Translating lyrics...")
        translated_segments = self.translator.translate_segments(segments, source_lang=detected_lang)

        # 4. Timestamp alignment
        logger.info("[4/5] Aligning timestamps...")
        aligned_segments = self.aligner.align(translated_segments)

        # 5. SRT generation
        srt_path = None
        if self.config.generate_srt:
            logger.info("[5/5] Generating SRT subtitle file...")
            srt_path = os.path.join(self.config.output_dir, f"{stem}_translated.srt")
            self.srt_gen.generate(aligned_segments, srt_path)
            logger.info(f"       SRT saved → {srt_path}")

        # 6. Metrics (optional)
        eval_metrics = {}
        if reference_lyrics:
            hypothesis = " ".join(s["text"] for s in segments)
            eval_metrics = self.metrics.compute_all(hypothesis, reference_lyrics, aligned_segments)

        result = PipelineResult(
            audio_path=audio_path,
            duration_seconds=duration,
            detected_language=detected_lang,
            detected_language_confidence=lang_prob,
            segments=segments,
            translated_segments=aligned_segments,
            srt_path=srt_path,
            processing_time=round(time.time() - t0, 2),
            metrics=eval_metrics,
        )
        logger.info(f"=== Done in {result.processing_time}s ===")
        return result


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Song Translation Pipeline")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default="base", help="Whisper model size")
    parser.add_argument("--device", default="cpu", help="cuda | cpu | mps")
    parser.add_argument("--backend", default="helsinki", help="Translation backend")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-srt", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(
        whisper_model=args.model,
        device=args.device,
        translation_backend=args.backend,
        output_dir=args.output_dir,
        generate_srt=not args.no_srt,
    )
    pipeline = SongTranslationPipeline(cfg)
    result = pipeline.run(args.audio)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Language detected : {result.detected_language} ({result.detected_language_confidence:.0%})")
    print(f"Duration          : {result.duration_seconds:.1f}s")
    print(f"Processing time   : {result.processing_time}s")
    if result.srt_path:
        print(f"SRT file          : {result.srt_path}")
    print("\nTranslated Lyrics:")
    print("-" * 60)
    for seg in result.translated_segments:
        print(f"[{seg['start']:.2f}s → {seg['end']:.2f}s]  {seg['translated_text']}")
