"""
Translation quality pipeline: tone control, false friends, grammar, cultural hints.
Integrates into the encoder-decoder translation flow for natural, meaning-first output.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tone/Register Control ──────────────────────────────────────────────

TONE_TAGS = {"FORMAL", "CASUAL", "NEUTRAL"}

TONE_PROMPTS = {
    "FORMAL": {
        "en": " [Formal register: use polite forms, avoid contractions, use full sentences]",
        "es": " [Registro formal: use usted, evite contracciones, oraciones completas]",
        "fr": " [Registre soutenu: utilisez vous, évitez les contractions, phrases complètes]",
        "de": " [Formelles Register: verwenden Sie Sie, keine Kontraktionen, vollständige Sätze]",
        "pt": " [Registro formal: use você, evite contrações, frases completas]",
        "it": " [Registro formale: usi Lei, eviti contrazioni, frasi complete]",
        "ja": " [敬体: ですます調を使用し、丁寧な表現]",
        "zh": " [正式语体: 使用礼貌用语，完整句子]",
        "ru": " [Официальный стиль: используйте Вы, полные предложения]",
        "ar": " [السجل الرسمي: استخدم صيغ المهذبة والجمل الكاملة]",
        "ko": " [격식체: 존댓말 사용, 완전한 문장]",
        "nl": " [Formeel register: gebruik u, vermijd samentrekkingen]",
        "pl": " [Oficjalny styl: używaj Pan/Pani, pełne zdania]",
        "tr": " [Resmi kayıt: nazlı ifadeler kullanın, tam cümleler]",
        "th": " [ทางการ: ใช้คำสุภาพ ประโยคสมบูรณ์]",
        "vi": " [Trang trọng: dùng kính ngữ, câu đầy đủ]",
        "hi": " [औपचारिक: आदरसूचक शब्दों का प्रयोग करें]",
        "sv": " [Formellt register: använd ni, fullständiga meningar]",
        "uk": " [Офіційний стиль: використовуйте Ви, повні речення]",
        "id": " [Formal: gunakan kata sopan, kalimat lengkap]",
    },
    "CASUAL": {
        "en": " [Casual register: use everyday language, contractions OK, conversational]",
        "es": " [Registro casual: lenguaje cotidiano, tuteo, conversacional]",
        "fr": " [Registre familier: langage quotidien, tutoiement]",
        "de": " [Umgangssprache: duzen, Alltagssprache]",
        "pt": " [Registro casual: linguagem do dia a dia, tratamento informal]",
        "it": " [Registro informale: dai del tu, linguaggio quotidiano]",
    },
}

FRIENDLY_TIE_BREAKER = {
    "en": " [Natural, idiomatic, meaning-first translation]",
}


def detect_tone(text: str) -> str:
    """Detect if text contains an explicit tone tag or infer from content."""
    text_stripped = text.strip().upper()
    for tag in TONE_TAGS:
        if text_stripped.startswith(f"[{tag}]"):
            return tag
    if text_stripped.startswith("[") and "]" in text_stripped[:20]:
        return "NEUTRAL"
    return "NEUTRAL"


def strip_tone_tag(text: str) -> str:
    """Remove tone tag prefix from text."""
    text_stripped = text.strip().upper()
    for tag in TONE_TAGS:
        marker = f"[{tag}]"
        if text_stripped.startswith(marker):
            return text[len(marker):].strip()
    return text


def apply_tone_prompt(text: str, tone: str, lang: str) -> str:
    """Append a tone/register hint to the source text."""
    if tone == "NEUTRAL":
        return text
    prompts = TONE_PROMPTS.get(tone, {})
    hint = prompts.get(lang, prompts.get("en", ""))
    if hint:
        return text + hint
    return text


# ── False Friends Dictionary ───────────────────────────────────────────

FALSE_FRIENDS: Dict[str, Dict[str, Dict[str, str]]] = {
    "en_es": {
        "embarazada": "pregnant (not embarrassed)",
        "sensible": "sensitive (not sensible)",
        "actualmente": "currently (not actually)",
        "asistir": "to attend (not to assist)",
        "bizarro": "dashing/brave (not bizarre)",
        "carpeta": "folder (not carpet)",
        "constipado": "to have a cold (not constipated)",
        "discutir": "to argue (not to discuss)",
        "en absoluto": "not at all (not absolutely)",
        "éxito": "success (not exit)",
        "fábrica": "factory (not fabric)",
        "grabar": "to record (not to grab)",
        "lectura": "reading (not lecture)",
        "molestar": "to bother (not to molest)",
        "noticia": "news (not notice)",
        "oficina": "office (not officina)",
        "pretender": "to try (not to pretend)",
        "realizar": "to do/carry out (not to realize)",
        "recordar": "to remember (not to record)",
        "sopa": "soup (not soap)",
        "trampa": "trap/trick (not tramp)",
    },
    "en_fr": {
        "actuellement": "currently (not actually)",
        "assister": "to attend (not to assist)",
        "blesser": "to wound (not to bless)",
        "coin": "corner (not coin)",
        "déception": "disappointment (not deception)",
        "défendre": "to forbid (not to defend)",
        "demander": "to ask (not to demand)",
        "éditeur": "publisher (not editor)",
        "envy": "annoyance (not envy)",
        "gift": "poison (not gift)",
        "ignorer": "to not know (not to ignore)",
        "journée": "day (not journey)",
        "librairie": "bookstore (not library)",
        "monnaie": "change/currency (not money)",
        "pain": "bread (not pain)",
        "pièce": "room/coin (not piece)",
        "propre": "clean (not proper)",
        "sensible": "sensitive (not sensible)",
        "sympathique": "nice (not sympathetic)",
        "veste": "jacket (not vest)",
    },
    "en_de": {
        "aktuell": "current (not actual)",
        "also": "so/therefore (not also)",
        "bald": "soon (not bald)",
        "bekommen": "to receive (not to become)",
        "billig": "cheap (not bill)",
        "brav": "well-behaved (not brave)",
        "Chef": "boss (not chef)",
        "eventuell": "possibly (not eventually)",
        "Gift": "poison (not gift)",
        "Handy": "mobile phone (not handy)",
        "Hell": "bright (not hell)",
        "herb": "bitter/tart (not herb)",
        "Mist": "manure (not mist)",
        "Muffe": "sleeve/pipe (not muff)",
        "Rat": "advice (not rat)",
        "sensibel": "sensitive (not sensible)",
        "Stift": "pen (not gift)",
        "winken": "to wave (not to wink)",
    },
}


def check_false_friends(text: str, source_lang: str, target_lang: str) -> List[Tuple[str, str, str]]:
    """Check source text for known false friends. Returns list of (word, warning, correct_meaning)."""
    pair_key = f"{source_lang}_{target_lang}"
    reverse_key = f"{target_lang}_{source_lang}"
    dict_key = pair_key if pair_key in FALSE_FRIENDS else reverse_key if reverse_key in FALSE_FRIENDS else None
    if not dict_key:
        return []
    ff_dict = FALSE_FRIENDS[dict_key]
    words = re.findall(r'\b\w+\b', text.lower())
    warnings = []
    for word in words:
        if word in ff_dict:
            warnings.append((word, f"False friend detected: '{word}' means '{ff_dict[word]}' in {target_lang}", ff_dict[word]))
    return warnings


# ── Quality Scoring ────────────────────────────────────────────────────


def score_translation_quality(
    source_text: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    log_probs: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Score a translation on multiple quality dimensions.

    Args:
        source_text: Original source text
        translation: Generated translation
        source_lang: Source language code
        target_lang: Target language code
        log_probs: Per-token log probabilities from decoder (if available)

    Returns:
        Dict of quality scores (0-1 scale, higher is better)
    """
    scores = {}

    # 1. Length ratio (penalize too short or too long)
    src_len = len(source_text.split())
    tgt_len = len(translation.split())
    if src_len > 0 and tgt_len > 0:
        ratio = tgt_len / src_len
        if 0.5 <= ratio <= 2.0:
            scores["length_score"] = 1.0 - abs(1.0 - ratio) / 1.5
        else:
            scores["length_score"] = 0.1
    else:
        scores["length_score"] = 0.0

    # 2. Perplexity (from log probs if available)
    if log_probs and len(log_probs) > 0:
        avg_log_prob = sum(log_probs) / len(log_probs)
        perplexity = 2.0 ** (-avg_log_prob)
        scores["perplexity"] = max(0.0, min(1.0, 1.0 - (perplexity / 100.0)))
    else:
        scores["perplexity"] = 0.5

    # 3. Repetition penalty
    words = translation.lower().split()
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        scores["diversity"] = min(1.0, unique_ratio * 1.5)
    else:
        scores["diversity"] = 1.0

    # 4. Overall quality
    scores["overall"] = sum(v for k, v in scores.items() if k != "overall") / max(len([k for k in scores if k != "overall"]), 1)

    return scores


# ── Grammar Post-Processing ────────────────────────────────────────────


def postprocess_grammar(text: str, lang: str) -> str:
    """Apply basic language-specific grammar fixes."""
    if lang == "en":
        text = re.sub(r'\bi\b', 'I', text)
        text = re.sub(r"n't ", "n't ", text)
        text = re.sub(r"' ([A-Za-z])", r"'\1", text)
    elif lang == "fr":
        text = re.sub(r'\bl\s+\'', "l'", text)
        text = re.sub(r'\bd\s+\'', "d'", text)
        text = re.sub(r'(\w)\s+-\s+(\w)', r'\1-\2', text)
    elif lang == "de":
        if text and text[0].islower() and not text.startswith(("kein", "mein", "sein", "ein")):
            text = text[0].upper() + text[1:]
    elif lang in ("es", "it", "pt"):
        text = re.sub(r'\b(\w+)lo\s+(\w+)\b', lambda m: m.group(1) + "lo " + m.group(2) if m.group(1).endswith(("r", "s")) else m.group(0), text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s([?.!,"])', r'\1', text)
    text = re.sub(r'(["\']) ', r'\1', text)
    return text


# ── Main quality pipeline ──────────────────────────────────────────────


class TranslationQualityPipeline:
    """Pipeline that wraps translation with quality enhancements.

    Features:
    - Tone/register detection and control via source prefix tags
    - False friends detection and logging
    - Beam reranking by quality score
    - Grammar post-processing
    - Domain-specific adapter routing
    """

    def __init__(self, false_friends_enabled: bool = True, grammar_postprocess: bool = True):
        self.false_friends_enabled = false_friends_enabled
        self.grammar_postprocess = grammar_postprocess

    def prepare_input(self, text: str, source_lang: str, target_lang: str, domain: Optional[str] = None) -> Tuple[str, str, str]:
        """Prepare text for translation: detect tone, check false friends.

        Returns:
            (processed_text, tone, domain)
        """
        tone = detect_tone(text)
        cleaned_text = strip_tone_tag(text)
        text_with_hints = apply_tone_prompt(cleaned_text, tone, source_lang)
        if tone == "NEUTRAL":
            text_with_hints = cleaned_text + FRIENDLY_TIE_BREAKER.get(source_lang, "")

        if self.false_friends_enabled:
            warnings = check_false_friends(cleaned_text, source_lang, target_lang)
            for word, warning, correct in warnings:
                logger.info(f"False friend: '{word}' → {correct}")

        return text_with_hints, tone, domain or "general"

    def postprocess(self, translation: str, target_lang: str, tone: str) -> str:
        """Post-process translation: grammar + tone consistency."""
        if self.grammar_postprocess:
            translation = postprocess_grammar(translation, target_lang)
        return translation

    def rerank_candidates(
        self,
        source_text: str,
        candidates: List[Tuple[str, Optional[List[float]]]],
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Rerank multiple translation candidates by quality score.

        Args:
            candidates: List of (translation, log_probs) tuples
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Best translation candidate
        """
        if len(candidates) <= 1:
            return candidates[0][0] if candidates else ""

        scored = []
        for translation, log_probs in candidates:
            scores = score_translation_quality(source_text, translation, source_lang, target_lang, log_probs)
            scored.append((scores.get("overall", 0.5), translation))

        scored.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"Reranked {len(candidates)} candidates, best score: {scored[0][0]:.3f}")
        return scored[0][1]
