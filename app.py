"""
Streamlit UI - Audio2Lyrics AI
Upload audio -> Transcribe -> Detect Language -> Translate -> Display + Download SRT
"""

import os
import sys
import gc
import time
import tempfile
from pathlib import Path

# FIX: flat repo layout - all .py files are at root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

try:
    from pipeline import SongTranslationPipeline, PipelineConfig
    from srt_generator import SRTGenerator
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

st.set_page_config(
    page_title="Audio2Lyrics AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .hero {
        background: linear-gradient(135deg, #0f0f23 0%, #1a0533 50%, #0d1b2a 100%);
        border-radius: 16px; padding: 2.5rem; text-align: center;
        margin-bottom: 2rem; border: 1px solid rgba(139, 92, 246, 0.3);
    }
    .hero h1 { color: #e9d5ff; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero p { color: #a78bfa; margin-top: 0.5rem; font-size: 1rem; }
    .metric-card {
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 12px; padding: 1rem 1.4rem; text-align: center;
    }
    .metric-card .label { color: #a78bfa; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #f3e8ff; font-size: 1.5rem; font-weight: 700; }
    .lang-badge {
        display: inline-block; background: rgba(139,92,246,0.15);
        border: 1px solid rgba(139,92,246,0.3); border-radius: 6px;
        padding: 2px 8px; font-size: 0.75rem; color: #c4b5fd;
        margin-right: 6px; font-family: monospace;
    }
    .segment-row {
        display: flex; gap: 1rem; padding: 0.75rem 0;
        border-bottom: 1px solid rgba(139, 92, 246, 0.12); align-items: flex-start;
    }
    .seg-time { color: #7c3aed; font-size: 0.78rem; font-family: monospace; min-width: 130px; padding-top: 2px; }
    .seg-orig { color: #c4b5fd; flex: 1; font-size: 0.9rem; }
    .seg-trans { color: #f3e8ff; flex: 1; font-size: 0.9rem; font-weight: 500; }
    .mixed-lang-banner {
        background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3);
        border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1rem;
        color: #fcd34d; font-size: 0.9rem;
    }
    .warn-banner {
        background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3);
        border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1rem;
        color: #fca5a5; font-size: 0.9rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
        width: 100%; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #a855f7); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎵 Audio2Lyrics AI</h1>
    <p>Audio &rarr; Lyrics &rarr; English Translation &middot; Timestamps &middot; Subtitles &middot; Mixed-language support</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")

    model_size = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=1,
        help="base = fastest on free tier. small = better accuracy but slower.",
    )
    device = st.selectbox("Device", ["cpu", "cuda", "mps"], index=0)
    backend = st.selectbox(
        "Translation Backend",
        ["helsinki", "openai", "google", "deepl"],
        help="Helsinki = free and fully offline after first download.",
    )
    handle_slang = st.toggle("Handle Slang", value=True)

    st.divider()
    st.subheader("Subtitle Options")
    gen_srt = st.toggle("Generate SRT File", value=True)
    gen_bilingual = st.toggle(
        "Bilingual SRT",
        value=False,
        help="Original + English stacked. Great for mixed-language songs.",
    )

    st.divider()
    st.caption("Supported: MP3 · WAV · FLAC · OGG · M4A · AAC")
    st.caption("Best results: songs under 5 minutes")

uploaded = st.file_uploader(
    "Upload your song audio file",
    type=["mp3", "wav", "flac", "ogg", "m4a", "aac"],
    label_visibility="collapsed",
)

if uploaded:
    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > 50:
        st.markdown(
            f'<div class="warn-banner">Large file ({size_mb:.0f}MB) detected. '
            f'Songs over 5 minutes may timeout on the free Streamlit tier. '
            f'Try trimming to under 5 minutes for best results.</div>',
            unsafe_allow_html=True,
        )

    st.audio(uploaded, format=uploaded.type)

    col_run, _ = st.columns([1, 3])
    with col_run:
        run_btn = st.button("Translate Song", use_container_width=True)

    if run_btn:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        out_dir = tempfile.mkdtemp()

        cfg = PipelineConfig(
            whisper_model=model_size,
            device=device,
            translation_backend=backend,
            handle_slang=handle_slang,
            generate_srt=gen_srt,
            output_dir=out_dir,
        )

        progress = st.progress(0, text="Initialising pipeline...")

        try:
            pipeline = SongTranslationPipeline(cfg)

            steps = [
                (15, "Preprocessing audio..."),
                (35, "Running Whisper ASR..."),
                (58, "Detecting language per segment..."),
                (75, "Translating lyrics..."),
                (90, "Aligning timestamps..."),
                (97, "Generating SRT..."),
            ]
            for pct, msg in steps:
                progress.progress(pct, text=msg)
                time.sleep(0.04)

            result = pipeline.run(tmp_path)

            bilingual_path = None
            if gen_srt and gen_bilingual and result.translated_segments:
                bilingual_path = os.path.join(
                    out_dir,
                    f"{Path(uploaded.name).stem}_bilingual.srt",
                )
                SRTGenerator().generate_bilingual(result.translated_segments, bilingual_path)

            progress.progress(100, text="Done!")
            time.sleep(0.3)
            progress.empty()

        except Exception as e:
            progress.empty()
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            st.stop()
        finally:
            # Always free memory - prevents Streamlit Cloud crashes on long songs
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        langs_found = set()
        for seg in result.translated_segments:
            lg = seg.get("segment_language") or result.detected_language
            langs_found.add(lg)

        if len(langs_found) > 1:
            lang_list = " + ".join(sorted(langs_found))
            st.markdown(
                f'<div class="mixed-lang-banner">Mixed-language song detected: {lang_list} '
                f'- each segment translated with its own model</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        lang_labels = {
            "es": "Spanish", "fr": "French", "de": "German",
            "pt": "Portuguese", "it": "Italian", "ja": "Japanese",
            "zh": "Chinese", "ko": "Korean", "ar": "Arabic",
            "hi": "Hindi", "ru": "Russian", "tr": "Turkish",
        }
        if len(langs_found) > 1:
            lang_display = "Mixed"
        else:
            lang_display = lang_labels.get(result.detected_language, result.detected_language.upper())

        metrics = [
            ("Language", lang_display),
            ("Confidence", f"{result.detected_language_confidence:.0%}"),
            ("Duration", f"{result.duration_seconds:.0f}s"),
            ("Process Time", f"{result.processing_time}s"),
        ]
        for col, (label, value) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{label}</div>'
                    f'<div class="value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        col_orig, col_trans = st.columns(2)
        with col_orig:
            st.subheader("Original Lyrics")
        with col_trans:
            st.subheader("English Translation")

        html_rows = []
        for seg in result.translated_segments:
            t = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"
            lang = seg.get("segment_language", result.detected_language).upper()
            orig = seg.get("original_text", seg.get("text", ""))
            trans = seg.get("translated_text", "")
            lang_badge = f'<span class="lang-badge">{lang}</span>' if len(langs_found) > 1 else ""
            html_rows.append(
                f'<div class="segment-row">'
                f'<div class="seg-time">{t}</div>'
                f'<div class="seg-orig">{lang_badge}{orig}</div>'
                f'<div class="seg-trans">{trans}</div>'
                f'</div>'
            )
        st.markdown("".join(html_rows), unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Downloads")

        has_bilingual = gen_srt and gen_bilingual and bilingual_path and os.path.exists(bilingual_path)
        dl_cols = st.columns(4 if has_bilingual else 3)

        with dl_cols[0]:
            transcript_text = "\n\n".join(
                f"[{s['start']:.2f}s - {s['end']:.2f}s]\n"
                f"Language : {s.get('segment_language', result.detected_language).upper()}\n"
                f"Original : {s.get('original_text', s.get('text', ''))}\n"
                f"English  : {s.get('translated_text', '')}"
                for s in result.translated_segments
            )
            st.download_button(
                "Transcript (.txt)",
                transcript_text,
                file_name=f"{Path(uploaded.name).stem}_transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with dl_cols[1]:
            if gen_srt and result.srt_path and os.path.exists(result.srt_path):
                srt_content = open(result.srt_path, "r", encoding="utf-8").read()
                st.download_button(
                    "SRT Subtitles",
                    srt_content,
                    file_name=f"{Path(uploaded.name).stem}_translated.srt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with dl_cols[2]:
            if has_bilingual:
                bil_content = open(bilingual_path, "r", encoding="utf-8").read()
                st.download_button(
                    "Bilingual SRT",
                    bil_content,
                    file_name=f"{Path(uploaded.name).stem}_bilingual.srt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with dl_cols[-1]:
            import json
            json_data = json.dumps({
                "audio": uploaded.name,
                "language": result.detected_language,
                "mixed_language": len(langs_found) > 1,
                "languages_detected": list(langs_found),
                "segments": result.translated_segments,
            }, indent=2)
            st.download_button(
                "JSON Data",
                json_data,
                file_name=f"{Path(uploaded.name).stem}_data.json",
                mime="application/json",
                use_container_width=True,
            )

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #7c3aed;
                border: 2px dashed rgba(139,92,246,0.3); border-radius:16px;">
        <p style="font-size:3rem; margin:0;">🎵</p>
        <p style="font-size:1.1rem;">Drop an audio file above to get started</p>
        <p style="font-size:0.85rem; color:#a78bfa;">MP3 · WAV · FLAC · OGG · M4A · AAC</p>
        <p style="font-size:0.85rem; color:#7c3aed;">Handles mixed-language songs (Hindi + Spanish etc.)</p>
        <p style="font-size:0.8rem; color:#6d28d9;">Best results: songs under 5 minutes</p>
    </div>
    """, unsafe_allow_html=True)
