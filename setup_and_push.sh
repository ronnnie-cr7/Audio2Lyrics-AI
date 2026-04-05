#!/usr/bin/env bash
# =============================================================================
# setup_and_push.sh — One-shot setup + GitHub push for Song Translator
# Usage: bash scripts/setup_and_push.sh
# =============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║    🎵  Song Translator — Setup & GitHub Push         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Check prerequisites ───────────────────────────────────────────────────
command -v git  >/dev/null 2>&1 || { echo "❌  git not found. Install git first."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌  python3 not found."; exit 1; }

echo "✅  Prerequisites OK"

# ── 2. Python virtual environment ───────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "📦  Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q

echo "📦  Installing dependencies..."
pip install -r requirements.txt -q
echo "✅  Dependencies installed"

# ── 3. Git init ──────────────────────────────────────────────────────────────
if [ ! -d ".git" ]; then
    echo "🔧  Initialising git repo..."
    git init
    git branch -M main
fi

# ── 4. Prompt for GitHub info ────────────────────────────────────────────────
echo ""
read -p "🔑  Enter your GitHub username: " GH_USER
read -p "📁  Enter repo name [song-translator]: " REPO_NAME
REPO_NAME=${REPO_NAME:-song-translator}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Creating remote repo: github.com/$GH_USER/$REPO_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 5. Create GitHub repo via API (requires gh CLI or token) ─────────────────
if command -v gh >/dev/null 2>&1; then
    gh repo create "$REPO_NAME" --public --description "🎵 Automatic Song Translation: Audio → Lyrics → English" --confirm 2>/dev/null || true
    echo "✅  GitHub repo created via gh CLI"
else
    echo "⚠️  GitHub CLI (gh) not installed."
    echo "    Create the repo manually at: https://github.com/new"
    echo "    Name it: $REPO_NAME"
    echo ""
    read -p "    Press ENTER once repo is created..."
fi

# ── 6. Commit and push ───────────────────────────────────────────────────────
git add -A
git commit -m "🎵 feat: initial commit — complete song translation pipeline

- AudioPreprocessor: 16kHz resampling, denoising, vocal separation
- WhisperASR: large-v3, word-level timestamps, VAD filter
- LyricsTranslator: MarianMT/DeepL/OpenAI + SlangNormalizer (30+ mappings)
- TimestampAligner: merge/split segments for readable subtitles
- SRTGenerator: standard + bilingual subtitle output
- EvaluationMetrics: WER, BLEU, chrF
- Streamlit UI: upload, transcribe, translate, download SRT
- Batch processor & CI/CD workflow included
- Full test suite (pytest)"

REMOTE_URL="https://github.com/$GH_USER/$REPO_NAME.git"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
git push -u origin main

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅  Successfully pushed!                            ║"
echo "║                                                      ║"
echo "║  🔗  https://github.com/$GH_USER/$REPO_NAME"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  streamlit run ui/app.py              # Launch UI"
echo "  python src/pipeline.py song.mp3     # CLI"
echo "  pytest tests/ -v                     # Run tests"
echo ""
