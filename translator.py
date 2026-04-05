"""
Lyrics Translation Module
- Primary: Helsinki-NLP MarianMT (offline, free)
- Fallback: DeepL / Google / OpenAI
- Slang normalisation layer (pre-processing)
- Post-processing: punctuation restoration, capitalisation
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slang / informal → standard mapping (Spanish-focused, extensible)
# ---------------------------------------------------------------------------
SLANG_MAP = {
    # Reggaeton / Latin trap
    "bichota": "boss woman",
    "tiguere": "street-smart person",
    "perrear": "dance",
    "yonki": "junkie",
    "janguear": "hang out",
    "cuero": "beautiful woman",
    "morbo": "desire",
    "wiri wiri": "non-stop / relentlessly",
    "chimba": "cool / awesome",
    "parce": "friend / bro",
    "ñero": "buddy / friend",
    "pana": "friend",
    "vaina": "thing / stuff",
    "mamagüebo": "fool",
    "cabrón": "badass / bastard",
    "güey": "dude",
    "chale": "no way / damn",
    "orale": "alright / okay",
    "órale": "alright / okay",
    "carnal": "brother / close friend",
    "jefa": "mom / boss",
    "simón": "yeah / yes",
    "chido": "cool",
    "neta": "truth / for real",
    "mamacita": "attractive woman",
    "papi": "daddy / attractive man",
    "mami": "mommy / attractive woman",
    # Anglicisms in Spanish rap
    "flow": "flow / style",
    "freestyle": "freestyle rap",
    "trap": "trap music",
    # Common contractions
    "pa'": "para (for)",
    "pa": "for",
    "na'": "nada (nothing)",
    "to'": "todo (everything)",
    "tó": "todo (everything)",
}


class SlangNormalizer:
    """Normalises slang terms before or after translation."""

    def __init__(self, slang_map: Optional[Dict] = None):
        self.map = {k.lower(): v for k, v in (slang_map or SLANG_MAP).items()}

    def normalize(self, text: str) -> str:
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            # Try bigrams first
            if i + 1 < len(words):
                bigram = (words[i] + " " + words[i + 1]).lower().strip(".,!?")
                if bigram in self.map:
                    result.append(self.map[bigram])
                    i += 2
                    continue
            token = words[i].lower().strip(".,!?\"'")
            if token in self.map:
                result.append(self.map[token])
            else:
                result.append(words[i])
            i += 1
        return " ".join(result)


# ---------------------------------------------------------------------------
class LyricsTranslator:
    """
    Translate timestamped segments into English.
    Supports multiple backends.
    """

    SUPPORTED_BACKENDS = ("helsinki", "deepl", "google", "openai")

    def __init__(self, backend: str = "helsinki", target_lang: str = "en", handle_slang: bool = True):
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {self.SUPPORTED_BACKENDS}")
        self.backend = backend
        self.target_lang = target_lang
        self.handle_slang = handle_slang
        self.slang = SlangNormalizer()
        self._models: Dict = {}

    # ------------------------------------------------------------------
    def translate_segments(self, segments: List[Dict], source_lang: str = "es") -> List[Dict]:
        """
        Add `translated_text` key to each segment dict (in-place copy).
        """
        results = []
        for seg in segments:
            raw = seg["text"]
            if self.handle_slang:
                raw = self.slang.normalize(raw)
            translated = self._translate(raw, source_lang)
            translated = self._postprocess(translated)
            results.append({**seg, "translated_text": translated, "original_text": seg["text"]})
        return results

    # ------------------------------------------------------------------
    def _translate(self, text: str, src: str) -> str:
        if not text.strip():
            return ""
        if self.backend == "helsinki":
            return self._helsinki(text, src)
        elif self.backend == "deepl":
            return self._deepl(text)
        elif self.backend == "google":
            return self._google(text)
        elif self.backend == "openai":
            return self._openai(text, src)

    # ------------------------------------------------------------------
    def _helsinki(self, text: str, src: str) -> str:
        """MarianMT via HuggingFace – fully offline after first download."""
        model_key = src
        if model_key not in self._models:
            from transformers import MarianMTModel, MarianTokenizer
            # Map ISO 639-1 → Helsinki model name
            lang_map = {
                "es": "Helsinki-NLP/opus-mt-es-en",
                "fr": "Helsinki-NLP/opus-mt-fr-en",
                "de": "Helsinki-NLP/opus-mt-de-en",
                "pt": "Helsinki-NLP/opus-mt-ROMANCE-en",
                "it": "Helsinki-NLP/opus-mt-it-en",
                "nl": "Helsinki-NLP/opus-mt-nl-en",
                "ru": "Helsinki-NLP/opus-mt-ru-en",
                "zh": "Helsinki-NLP/opus-mt-zh-en",
                "ja": "Helsinki-NLP/opus-mt-ja-en",
                "ko": "Helsinki-NLP/opus-mt-ko-en",
                "ar": "Helsinki-NLP/opus-mt-ar-en",
            }
            model_name = lang_map.get(src, f"Helsinki-NLP/opus-mt-{src}-en")
            logger.info(f"  Loading MarianMT model: {model_name}")
            tok = MarianTokenizer.from_pretrained(model_name)
            mdl = MarianMTModel.from_pretrained(model_name)
            self._models[model_key] = (tok, mdl)

        tokenizer, model = self._models[model_key]
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs, num_beams=4, early_stopping=True)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def _deepl(self, text: str) -> str:
        import deepl
        key = os.environ.get("DEEPL_API_KEY", "")
        translator = deepl.Translator(key)
        result = translator.translate_text(text, target_lang="EN-US")
        return result.text

    def _google(self, text: str) -> str:
        from google.cloud import translate_v2 as translate
        client = translate.Client()
        result = client.translate(text, target_language="en")
        return result["translatedText"]

    def _openai(self, text: str, src: str) -> str:
        import openai
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a professional song lyrics translator. "
                    "Preserve the poetic style, rhythm feel, and cultural nuance. "
                    "Output ONLY the English translation, nothing else."
                )},
                {"role": "user", "content": f"Translate these song lyrics from {src} to English:\n\n{text}"},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    @staticmethod
    def _postprocess(text: str) -> str:
        """Light cleanup: fix spacing, capitalise first word."""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s([.,!?;:])", r"\1", text)
        if text:
            text = text[0].upper() + text[1:]
        return text

    # ------------------------------------------------------------------
    @property
    def supported_languages(self) -> List[str]:
        return ["es", "fr", "de", "pt", "it", "nl", "ru", "zh", "ja", "ko", "ar"]
