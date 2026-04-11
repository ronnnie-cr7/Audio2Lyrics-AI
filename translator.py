"""
Lyrics Translation Module
- Per-segment language detection (handles mixed-language songs)
- Primary: Helsinki-NLP MarianMT (offline, free)
- Slang normalisation layer
- Post-processing
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

SLANG_MAP = {
    "bichota": "boss woman",
    "tiguere": "street-smart person",
    "perrear": "dance",
    "yonki": "junkie",
    "janguear": "hang out",
    "cuero": "beautiful woman",
    "morbo": "desire",
    "wiri wiri": "non-stop",
    "chimba": "cool",
    "parce": "friend",
    "nero": "buddy",
    "pana": "friend",
    "vaina": "thing",
    "cabron": "badass",
    "guey": "dude",
    "chale": "no way",
    "orale": "alright",
    "carnal": "brother",
    "jefa": "mom / boss",
    "simon": "yes",
    "chido": "cool",
    "neta": "for real",
    "mamacita": "attractive woman",
    "papi": "attractive man",
    "mami": "attractive woman",
    "yaar": "friend",
    "dil": "heart",
    "ishq": "love",
    "mohabbat": "love",
    "zindagi": "life",
    "tere bina": "without you",
    "mera dil": "my heart",
    "pyaar": "love",
    "dost": "friend",
    "pagal": "crazy",
    "bindaas": "carefree",
    "pa'": "for",
    "na'": "nothing",
    "to'": "everything",
}

# Languages that map directly to a working Helsinki model
HELSINKI_MODELS = {
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
    "tr": "Helsinki-NLP/opus-mt-tr-en",
    "pl": "Helsinki-NLP/opus-mt-pl-en",
    "uk": "Helsinki-NLP/opus-mt-uk-en",
    "sv": "Helsinki-NLP/opus-mt-sv-en",
    # All others fall back to multilingual
    "hi": "Helsinki-NLP/opus-mt-mul-en",
    "sw": "Helsinki-NLP/opus-mt-mul-en",
    "tl": "Helsinki-NLP/opus-mt-mul-en",
    "ms": "Helsinki-NLP/opus-mt-mul-en",
    "bn": "Helsinki-NLP/opus-mt-mul-en",
    "ta": "Helsinki-NLP/opus-mt-mul-en",
    "id": "Helsinki-NLP/opus-mt-id-en",
}

MULTILINGUAL_FALLBACK = "Helsinki-NLP/opus-mt-mul-en"


def detect_language(text: str) -> Optional[str]:
    if not text or len(text.strip()) < 8:
        return None
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return None


class SlangNormalizer:
    def __init__(self, slang_map: Optional[Dict] = None):
        self.map = {k.lower(): v for k, v in (slang_map or SLANG_MAP).items()}

    def normalize(self, text: str) -> str:
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            if i + 1 < len(words):
                bigram = (words[i] + " " + words[i + 1]).lower().strip(".,!?")
                if bigram in self.map:
                    result.append(self.map[bigram])
                    i += 2
                    continue
            token = words[i].lower().strip(".,!?\"'")
            result.append(self.map[token] if token in self.map else words[i])
            i += 1
        return " ".join(result)


class LyricsTranslator:

    SUPPORTED_BACKENDS = ("helsinki", "deepl", "google", "openai")

    def __init__(self, backend: str = "helsinki", target_lang: str = "en", handle_slang: bool = True):
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {self.SUPPORTED_BACKENDS}")
        self.backend = backend
        self.target_lang = target_lang
        self.handle_slang = handle_slang
        self.slang = SlangNormalizer()
        self._models: Dict = {}

    def translate_segments(self, segments: List[Dict], source_lang: str = "es") -> List[Dict]:
        results = []
        for seg in segments:
            raw = seg["text"]
            seg_lang = detect_language(raw) or source_lang

            if seg_lang == self.target_lang:
                results.append({
                    **seg,
                    "original_text": raw,
                    "translated_text": raw,
                    "segment_language": seg_lang,
                })
                continue

            clean = self.slang.normalize(raw) if self.handle_slang else raw

            try:
                translated = self._translate(clean, seg_lang)
                translated = self._postprocess(translated)
            except Exception as e:
                logger.warning(f"Translation failed for lang={seg_lang}: {e}, returning original")
                translated = raw

            results.append({
                **seg,
                "original_text": raw,
                "translated_text": translated,
                "segment_language": seg_lang,
            })
        return results

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
        return text

    def _helsinki(self, text: str, src: str) -> str:
        if src not in self._models:
            from transformers import MarianMTModel, MarianTokenizer
            model_name = HELSINKI_MODELS.get(src, MULTILINGUAL_FALLBACK)
            try:
                logger.info(f"  Loading MarianMT: {model_name}")
                tok = MarianTokenizer.from_pretrained(model_name)
                mdl = MarianMTModel.from_pretrained(model_name)
            except Exception:
                logger.warning(f"  {model_name} failed, using multilingual fallback")
                tok = MarianTokenizer.from_pretrained(MULTILINGUAL_FALLBACK)
                mdl = MarianMTModel.from_pretrained(MULTILINGUAL_FALLBACK)
            self._models[src] = (tok, mdl)

        tokenizer, model = self._models[src]
        inputs = tokenizer([text], return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        translated = model.generate(**inputs, num_beams=4, early_stopping=True)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def _deepl(self, text: str) -> str:
        import os
        import deepl
        translator = deepl.Translator(os.environ["DEEPL_API_KEY"])
        return translator.translate_text(text, target_lang="EN-US").text

    def _google(self, text: str) -> str:
        from google.cloud import translate_v2 as translate
        return translate.Client().translate(text, target_language="en")["translatedText"]

    def _openai(self, text: str, src: str) -> str:
        import os
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional song lyrics translator. Preserve poetic style. Output ONLY the English translation."},
                {"role": "user", "content": f"Translate from {src} to English:\n\n{text}"},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _postprocess(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s([.,!?;:])", r"\1", text)
        return text[0].upper() + text[1:] if text else text
