# 🎵 Automatic Song Translation System

> **Audio → Lyrics → English Translation with Timestamps & Subtitles**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Whisper](https://img.shields.io/badge/ASR-Whisper%20large--v3-green.svg)](https://github.com/openai/whisper)
[![MarianMT](https://img.shields.io/badge/Translation-MarianMT-orange.svg)](https://huggingface.co/Helsinki-NLP)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     SONG TRANSLATION PIPELINE                            │
├──────────────┬─────────────────┬──────────────┬──────────────┬──────────┤
│  INPUT AUDIO │  PREPROCESSING  │  WHISPER ASR │ TRANSLATION  │  OUTPUT  │
│              │                 │              │              │          │
│  MP3/WAV/    │  • Resample     │  • Language  │  • MarianMT  │  • SRT   │
│  FLAC/OGG/   │    → 16kHz mono │    detection │  • DeepL     │  • JSON  │
│  M4A/AAC     │  • Normalise    │  • Word-lvl  │  • OpenAI    │  • TXT   │
│              │  • Denoise      │    timestamps│  • Google    │  • UI    │
│              │  • Vocal sep.   │  • VAD filter│  • Slang     │          │
│              │    (optional)   │              │    handling  │          │
└──────────────┴─────────────────┴──────────────┴──────────────┴──────────┘
```

### Data Flow
```
Song File
   │
   ▼
AudioPreprocessor ──► 16kHz WAV (denoised, normalised)
   │
   ▼
WhisperASR ──► Segments [{start, end, text, words, lang}]
   │
   ▼
SlangNormalizer ──► Cleaned text (bichota → boss woman)
   │
   ▼
LyricsTranslator ──► [{..., translated_text}]
   │
   ▼
TimestampAligner ──► Merged / split for readability
   │
   ▼
SRTGenerator ──► .srt subtitle file
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/song-translator.git
cd song-translator

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (required by pydub / whisper)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
# Windows: download from https://ffmpeg.org/download.html
```

### 2. Run the CLI

```bash
# Basic usage
python src/pipeline.py path/to/song.mp3

# With options
python src/pipeline.py path/to/song.mp3 \
    --model large-v3 \
    --device cuda \
    --backend helsinki \
    --output-dir outputs/
```

### 3. Launch the Streamlit UI

```bash
streamlit run ui/app.py
# Opens at http://localhost:8501
```

---

## 📁 Project Structure

```
song-translator/
├── src/
│   ├── pipeline.py               # Main orchestrator
│   ├── audio/
│   │   └── preprocessor.py       # Audio loading, resampling, denoising
│   ├── asr/
│   │   └── whisper_asr.py        # Whisper transcription + language detection
│   ├── translation/
│   │   └── translator.py         # MarianMT / DeepL / OpenAI + slang handling
│   ├── alignment/
│   │   └── aligner.py            # Timestamp alignment, merge/split logic
│   └── utils/
│       ├── srt_generator.py      # SRT subtitle file creation
│       └── metrics.py            # WER, BLEU, chrF evaluation
├── ui/
│   └── app.py                    # Streamlit web interface
├── data/                         # Sample audio & reference lyrics
├── outputs/                      # Generated SRT / transcripts
├── tests/
│   └── test_pipeline.py          # Unit & integration tests
├── scripts/
│   └── batch_process.py          # Process multiple songs at once
├── docs/
│   └── architecture.md           # Detailed architecture notes
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Configuration

### PipelineConfig Options

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `whisper_model` | `large-v3` | `tiny/base/small/medium/large-v2/large-v3` | ASR accuracy vs speed |
| `device` | `cuda` | `cuda/cpu/mps` | Inference device |
| `compute_type` | `float16` | `float16/int8` | Quantisation |
| `translation_backend` | `helsinki` | `helsinki/deepl/openai/google` | Translation engine |
| `handle_slang` | `True` | bool | Slang normalisation layer |
| `denoise` | `True` | bool | Spectral noise reduction |
| `remove_vocals` | `False` | bool | Demucs vocal isolation |
| `generate_srt` | `True` | bool | Output .srt file |

### Environment Variables (.env)

```bash
DEEPL_API_KEY=your_deepl_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
```

---

## 🌍 Supported Languages

| Language | Code | Helsinki Model |
|----------|------|---------------|
| Spanish | `es` | `opus-mt-es-en` |
| French | `fr` | `opus-mt-fr-en` |
| German | `de` | `opus-mt-de-en` |
| Portuguese | `pt` | `opus-mt-ROMANCE-en` |
| Italian | `it` | `opus-mt-it-en` |
| Japanese | `ja` | `opus-mt-ja-en` |
| Chinese | `zh` | `opus-mt-zh-en` |
| Korean | `ko` | `opus-mt-ko-en` |
| Arabic | `ar` | `opus-mt-ar-en` |
| Russian | `ru` | `opus-mt-ru-en` |

---

## 🎭 Slang Handling

The `SlangNormalizer` pre-processes text before translation using a curated mapping:

```python
"bichota"  → "boss woman"
"perrear"  → "dance"
"janguear" → "hang out"
"pa'"      → "para (for)"
"to'"      → "todo (everything)"
# ... 30+ mappings, easily extensible
```

Add custom slang in `src/translation/translator.py → SLANG_MAP`.

---

## 📊 Evaluation Metrics

| Metric | Description | Good Score |
|--------|-------------|------------|
| WER | Word Error Rate (ASR) | < 15% |
| BLEU | Translation n-gram overlap | > 30 |
| chrF | Character n-gram F-score | > 50 |

```bash
# Run with reference lyrics
python src/pipeline.py song.mp3 --reference-lyrics lyrics.txt
```

---

## 🔥 Advanced Features

### Vocal Separation (Demucs)
```bash
pip install demucs
# Then set remove_vocals=True in PipelineConfig
```

### Batch Processing
```bash
python scripts/batch_process.py --input-dir songs/ --output-dir outputs/
```

### Bilingual SRT
```python
from utils.srt_generator import SRTGenerator
SRTGenerator().generate_bilingual(segments, "output_bilingual.srt")
```

---

## 🎯 Real-World Extensions

- **Spotify / Apple Music**: Use their audio APIs + pipeline to show translated lyrics in sync
- **YouTube**: Process auto-captions → retranslate with better accuracy
- **Language Learning**: Build Duolingo-like apps using timestamped bilingual lyrics
- **Karaoke**: Generate coloured word-by-word subtitles synced to music

### Scalability
- **Containerise**: Docker + GPU (NVIDIA CUDA base image)
- **Queue**: Celery + Redis for async job processing
- **Storage**: S3 for audio, PostgreSQL for transcripts
- **API**: FastAPI wrapper around `SongTranslationPipeline.run()`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
