# 🎵 Audio2Lyrics AI

> Built a system that Spotify and Apple Music still don’t have.

Convert songs → lyrics → English translation → synced subtitles.

---

## 🚀 What this does

- 🎧 Extracts lyrics directly from audio (no text input needed)
- 🌍 Translates songs into English
- ⏱️ Keeps timestamps aligned with music
- 📄 Generates subtitle files (.srt)

Works even for fast, slang-heavy songs.

---

## 🧠 Tech Stack

- faster-whisper (speech recognition)
- NLP Translation (Helsinki / MarianMT)
- FFmpeg (audio processing)
- Streamlit (UI)

---

Audio File
   ↓
Preprocessing (FFmpeg)
   ↓
ASR (faster-whisper)
   ↓
Translation (NLP models)
   ↓
Alignment
   ↓
SRT Generation

## ⚡ Demo

Upload a song → click translate → get synced English lyrics.

🔗 Live App: https://audio2lyrics-ai.streamlit.app/

---

## 💻 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py

