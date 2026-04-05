"""
Streamlit UI — Automatic Song Translation System
Upload audio → Transcribe → Detect Language → Translate → Display + Download SRT
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import streamlit as st

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import SongTranslationPipeline, PipelineConfig

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🎵 Song Translator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0f0f23 0%, #1a0533 50%, #0d1b2a 100%);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    .hero h1 { color: #e9d5ff; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .hero p  { color: #a78bfa; margin-top: 0.5rem; font-size: 1.05rem; }

    .metric-card {
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        text-align: center;
    }
    .metric-card .label { color: #a78bfa; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #f3e8ff; font-size: 1.6rem; font-weight: 700; }

    .segment-row {
        display: flex;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(139, 92, 246, 0.12);
        align-items: flex-start;
    }
    .seg-time { color: #7c3aed; font-size: 0.78rem; font-family: monospace; min-width: 120px; padding-top: 2px; }
    .seg-orig { color: #c4b5fd; flex: 1; font-size: 0.9rem; }
    .seg-trans { color: #f3e8ff; flex: 1; font-size: 0.9rem; font-weight: 500; }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #a855f7); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎵 Automatic Song Translator</h1>
    <p>Audio → Lyrics → English Translation with timestamps & subtitles</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    model_size = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=2,
        help="Larger = more accurate, slower",
    )
    device = st.selectbox("Device", ["cpu", "cuda", "mps"], index=0)
    backend = st.selectbox(
        "Translation Backend",
        ["helsinki", "openai", "google", "deepl"],
        help="Helsinki = free & offline. Others need API keys.",
    )
    handle_slang = st.toggle("Handle Slang & Informal Language", value=True)
    gen_srt = st.toggle("Generate SRT Subtitle File", value=True)
    gen_bilingual = st.toggle("Bilingual SRT (Original + Translation)", value=False)

    st.divider()
    st.caption("Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC")
    st.caption("Best results: songs with clear vocals")

# ---------------------------------------------------------------------------
# Main upload area
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload your song audio file",
    type=["mp3", "wav", "flac", "ogg", "m4a", "aac"],
    label_visibility="collapsed",
)

if uploaded:
    st.audio(uploaded, format=uploaded.type)

    col_run, col_gap = st.columns([1, 3])
    with col_run:
        run_btn = st.button("🚀 Translate Song", use_container_width=True)

    if run_btn:
        # Save upload to temp file
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        out_dir = tempfile.mkdtemp()

        # Build config
        cfg = PipelineConfig(
            whisper_model=model_size,
            device=device,
            translation_backend=backend,
            handle_slang=handle_slang,
            generate_srt=gen_srt,
            output_dir=out_dir,
        )

        # Run pipeline with progress display
        progress = st.progress(0, text="Initialising pipeline...")
        status = st.empty()

        try:
            pipeline = SongTranslationPipeline(cfg)

            # Monkey-patch to show progress steps
            stages = [
                (10, "🔊 Preprocessing audio..."),
                (30, "🤖 Running Whisper ASR..."),
                (60, "🌐 Detecting language..."),
                (75, "✍️ Translating lyrics..."),
                (90, "⏱️ Aligning timestamps..."),
                (98, "📄 Generating SRT..."),
            ]

            for pct, msg in stages:
                progress.progress(pct, text=msg)
                time.sleep(0.05)

            result = pipeline.run(tmp_path)

            # Optionally generate bilingual SRT
            if gen_bilingual and result.srt_path:
                from utils.srt_generator import SRTGenerator
                bilingual_path = result.srt_path.replace(".srt", "_bilingual.srt")
                SRTGenerator().generate_bilingual(result.translated_segments, bilingual_path)
                result.srt_path = bilingual_path

            progress.progress(100, text="✅ Done!")
            time.sleep(0.5)
            progress.empty()
            status.empty()

        except Exception as e:
            progress.empty()
            st.error(f"Pipeline error: {e}")
            st.stop()

        # ---------------------------------------------------------------
        # Metrics Row
        # ---------------------------------------------------------------
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        lang_labels = {
            "es": "🇪🇸 Spanish", "fr": "🇫🇷 French", "de": "🇩🇪 German",
            "pt": "🇧🇷 Portuguese", "it": "🇮🇹 Italian", "ja": "🇯🇵 Japanese",
            "zh": "🇨🇳 Chinese", "ko": "🇰🇷 Korean", "ar": "🇸🇦 Arabic",
        }
        lang_display = lang_labels.get(result.detected_language, f"🌐 {result.detected_language.upper()}")
        with m1:
            st.markdown(f'<div class="metric-card"><div class="label">Language</div><div class="value">{lang_display}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="label">Confidence</div><div class="value">{result.detected_language_confidence:.0%}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="label">Duration</div><div class="value">{result.duration_seconds:.0f}s</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="label">Process Time</div><div class="value">{result.processing_time}s</div></div>', unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # Lyrics display
        # ---------------------------------------------------------------
        st.markdown("---")
        col_orig, col_trans = st.columns(2)
        with col_orig:
            st.subheader("📝 Original Lyrics")
        with col_trans:
            st.subheader("🇬🇧 English Translation")

        html_rows = []
        for seg in result.translated_segments:
            t = f"[{seg['start']:.1f}s → {seg['end']:.1f}s]"
            orig = seg.get("original_text", seg.get("text", ""))
            trans = seg.get("translated_text", "")
            html_rows.append(f"""
            <div class="segment-row">
                <div class="seg-time">{t}</div>
                <div class="seg-orig">{orig}</div>
                <div class="seg-trans">{trans}</div>
            </div>""")

        st.markdown("".join(html_rows), unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # Downloads
        # ---------------------------------------------------------------
        st.markdown("---")
        st.subheader("📥 Downloads")
        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            # Full transcript as text
            transcript_text = "\n\n".join(
                f"[{s['start']:.2f}s → {s['end']:.2f}s]\n"
                f"Original: {s.get('original_text', s.get('text',''))}\n"
                f"Translation: {s.get('translated_text','')}"
                for s in result.translated_segments
            )
            st.download_button(
                "📄 Download Transcript",
                transcript_text,
                file_name=f"{Path(uploaded.name).stem}_transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with dl2:
            if result.srt_path and os.path.exists(result.srt_path):
                with open(result.srt_path, "r", encoding="utf-8") as f:
                    srt_content = f.read()
                st.download_button(
                    "🎬 Download SRT Subtitles",
                    srt_content,
                    file_name=f"{Path(uploaded.name).stem}_translated.srt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with dl3:
            import json
            json_data = json.dumps({
                "audio": uploaded.name,
                "language": result.detected_language,
                "segments": result.translated_segments,
            }, indent=2)
            st.download_button(
                "📊 Download JSON",
                json_data,
                file_name=f"{Path(uploaded.name).stem}_data.json",
                mime="application/json",
                use_container_width=True,
            )

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #7c3aed; border: 2px dashed rgba(139,92,246,0.3); border-radius:16px;">
        <p style="font-size:3rem; margin:0;">🎵</p>
        <p style="font-size:1.1rem;">Drop an audio file above to get started</p>
        <p style="font-size:0.85rem; color:#a78bfa;">MP3 • WAV • FLAC • OGG • M4A • AAC</p>
    </div>
    """, unsafe_allow_html=True)
