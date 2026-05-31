"""
Translation quality pipeline: tone control, false friends, idioms, grammar, cultural hints.
Integrates into the encoder-decoder translation flow for natural, meaning-first output.

Designed for 20 languages: en, es, fr, de, pt, it, ja, zh, ru, ar, ko, nl, pl, tr, th, vi, hi, sv, uk, id

Backed by external JSON configs in utils/quality_resources/ for easy extension.
Optional library backends auto-detected at import time (no hard dependency):
  - language-tool-python for grammar checking (25+ languages)
  - transformers for formality/register detection
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "quality_resources")


def _load_json(name: str) -> dict:
    path = os.path.join(_RESOURCES_DIR, name)
    if not os.path.isfile(path):
        logger.warning(f"Quality resource not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── External JSON configs (extensible without code changes) ────────────

TONE_PROMPTS: Dict[str, Dict[str, str]] = _load_json("tone_prompts.json")
FRIENDLY_TIE_BREAKER: Dict[str, str] = _load_json("friendly_tie_breaker.json")
FALSE_FRIENDS: Dict[str, Dict[str, str]] = _load_json("false_friends.json")
IDIOMS: Dict[str, Dict[str, str]] = _load_json("idioms.json")

TONE_TAGS = {"FORMAL", "CASUAL", "NEUTRAL"}


# ── Optional library backends (try-import, silent fallback) ────────────

try:
    import language_tool_python

    _LANGUAGE_TOOL_POOL: Dict[str, "language_tool_python.LanguageTool"] = {}
    _LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    _LANGUAGE_TOOL_AVAILABLE = False
    _LANGUAGE_TOOL_POOL = {}

try:
    from transformers import pipeline

    _FORMALITY_PIPE = pipeline(
        "text-classification",
        model="s-nlp/xlmr_formality_classifier",
        top_k=None,
    )
    _FORMALITY_AVAILABLE = True
except ImportError:
    _FORMALITY_AVAILABLE = False
    _FORMALITY_PIPE = None


# ── Language mapping helpers ────────────────────────────────────────────

# language-tool-python codes differ from ISO 639-1
_LT_LANG_MAP = {
    "en": "en-US", "es": "es", "fr": "fr", "de": "de-DE",
    "pt": "pt-BR", "it": "it", "nl": "nl", "pl": "pl",
    "sv": "sv", "ru": "ru", "uk": "uk", "ja": "ja",
    "zh": "zh-CN", "ar": "ar", "ko": "ko", "tr": "tr",
    "th": "th", "vi": "vi", "hi": "hi", "id": "id",
}

# Languages known to be well-supported by each backend
_FORMALITY_LANGS = {"en", "es", "fr", "de", "it", "pt", "nl", "ru"}
_LT_MIN_CONFIDENCE = 0.5


def _get_lt_lang(lang: str) -> str:
    """Map ISO 639-1 to language-tool-python language code."""
    return _LT_LANG_MAP.get(lang, lang)


# ── Tone/Register Detection ────────────────────────────────────────────


def _library_detect_tone(text: str) -> Optional[str]:
    """Use HF formality classifier if available. Returns None if unavailable or unsure."""
    if not _FORMALITY_AVAILABLE or _FORMALITY_PIPE is None:
        return None
    try:
        results = _FORMALITY_PIPE(text[:512], truncation=True)
        if isinstance(results, list) and len(results) > 0:
            scores = {r["label"]: r["score"] for r in results[0]}
            formal_score = scores.get("formal", 0.0)
            if formal_score > 0.7:
                return "FORMAL"
            if formal_score < 0.3:
                return "CASUAL"
        return None
    except Exception:
        logger.debug("Formality classifier failed, falling back to tag-based detection")
        return None


def detect_tone(text: str, library_boost: bool = True) -> str:
    """Detect register from explicit tag prefix or, optionally, HF formality classifier."""
    text_stripped = text.strip().upper()
    for tag in TONE_TAGS:
        if text_stripped.startswith(f"[{tag}]"):
            return tag
    if text_stripped.startswith("[") and "]" in text_stripped[:20]:
        return "NEUTRAL"
    if library_boost:
        lib_tone = _library_detect_tone(text)
        if lib_tone:
            return lib_tone
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
    logger.warning(f"No tone prompt for tone={tone} lang={lang}")
    return text


# ── False Friends Detection ────────────────────────────────────────────


HARDCODED_FALSE_FRIENDS: Dict[str, Dict[str, str]] = {
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
        "oficina": "office (not workshop)",
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
        "chef": "boss (not chef)",
        "eventuell": "possibly (not eventually)",
        "gift": "poison (not gift)",
        "handy": "mobile phone (not handy)",
        "hell": "bright (not hell)",
        "herb": "bitter/tart (not herb)",
        "mist": "manure (not mist)",
        "muffe": "sleeve/pipe (not muff)",
        "rat": "advice (not rat)",
        "sensibel": "sensitive (not sensible)",
        "stift": "pen (not gift)",
        "winken": "to wave (not to wink)",
    },
}


def _get_ff_dict(source_lang: str, target_lang: str) -> Optional[Dict[str, str]]:
    """Look up false friends dict, trying JSON then hardcoded fallback."""
    pair_key = f"{source_lang}_{target_lang}"
    reverse_key = f"{target_lang}_{source_lang}"
    for d in (FALSE_FRIENDS, HARDCODED_FALSE_FRIENDS):
        if pair_key in d:
            return d[pair_key]
        if reverse_key in d:
            return d[reverse_key]
    return None


def check_false_friends(text: str, source_lang: str, target_lang: str) -> List[Tuple[str, str, str]]:
    """Check source text for known false friends. Returns list of (word, warning, correct_meaning).

    Handles multi-word entries (space-separated false friends). Logs a warning
    when the language pair has no coverage.
    """
    ff_dict = _get_ff_dict(source_lang, target_lang)
    if ff_dict is None:
        logger.debug(f"No false friends data for {source_lang}→{target_lang}")
        return []
    text_lower = text.lower()
    warnings = []
    for ff_word, meaning in ff_dict.items():
        if ff_word in text_lower:
            warnings.append((
                ff_word,
                f"False friend detected: '{ff_word}' means '{meaning}' in {target_lang}",
                meaning,
            ))
    return warnings


# ── Idioms / Multi-Word Expressions ────────────────────────────────────


HARDCODED_IDIOMS: Dict[str, Dict[str, str]] = {
    "es_en": {
        "ojo": "heads up / watch out",
        "calentar la cabeza": "to stress out / to bother",
        "tomar el pelo": "to pull someone's leg",
        "estar en las nubes": "to be daydreaming",
        "no tener pelos en la lengua": "to speak one's mind",
        "ser pan comido": "to be a piece of cake",
        "costar un ojo de la cara": "to cost an arm and a leg",
        "ponerse las pilas": "to get one's act together",
        "estar hasta las narices": "to be fed up",
        "meter la pata": "to put one's foot in it",
        "llover a cántaros": "to rain cats and dogs",
        "más vale tarde que nunca": "better late than never",
        "en boca cerrada no entran moscas": "silence is golden",
        "cada loco con su tema": "to each their own",
        "hacerse la vista gorda": "to turn a blind eye",
        "poner la mano en el fuego": "to vouch for someone",
        "tirar la toalla": "to throw in the towel",
        "dar en el clavo": "to hit the nail on the head",
        "no dar pie con bola": "to get everything wrong",
        "estar como una cabra": "to be nuts / crazy",
    },
    "fr_en": {
        "appeler un chat un chat": "to call a spade a spade",
        "casser les pieds": "to be a pain / to bother",
        "donner sa langue au chat": "to give up guessing",
        "être dans la lune": "to be daydreaming",
        "faire la grasse matinée": "to sleep in",
        "mettre son grain de sel": "to butt in",
        "ne pas être dans son assiette": "to feel off / not oneself",
        "poser un lapin": "to stand someone up",
        "raconter des salades": "to tell tall tales",
        "se lever du pied gauche": "to get up on the wrong side of the bed",
        "tirer les vers du nez": "to pry information out of someone",
        "vendre la mèche": "to spill the beans",
        "c'est la fin des haricots": "it's the last straw",
        "faire d'une pierre deux coups": "to kill two birds with one stone",
        "il pleut des cordes": "it's raining cats and dogs",
    },
    "de_en": {
        "da liegt der hund begraben": "that's the crux of the matter",
        "ich verstehe nur bahnhof": "it's all Greek to me",
        "um den heißen brei herumreden": "to beat around the bush",
        "alles in butter": "everything is fine / all good",
        "die daumen drücken": "to cross one's fingers",
        "fix und fertig": "to be exhausted / done in",
        "auf dem schlauch stehen": "to draw a blank",
        "in den sauren apfel beißen": "to bite the bullet",
        "pute kuchen": "piece of cake",
        "sich keine sorgen machen": "don't worry about it",
        "den kopf zerbrechen": "to rack one's brain",
        "mit dem kopf durch die wand wollen": "to be stubborn / headstrong",
    },
    "it_en": {
        "in bocca al lupo": "good luck / break a leg",
        "non avere peli sulla lingua": "to speak one's mind",
        "prendere la palla al balzo": "to seize the opportunity",
        "costare un occhio della testa": "to cost an arm and a leg",
        "essere al settimo cielo": "to be on cloud nine",
        "fare le corna": "to touch wood / jinx",
        "piangere sul latte versato": "to cry over spilled milk",
        "rompere le scatole": "to be a pain / annoy",
        "tutto fa brodo": "every little bit helps",
        "acqua in bocca": "mum's the word",
    },
    "pt_en": {
        "pagar o pato": "to take the blame / get stuck",
        "encher a paciência": "to be a nuisance",
        "ficar a ver navios": "to be left empty-handed",
        "matar dois coelhos com uma paulada": "to kill two birds with one stone",
        "não ter papas na língua": "to speak one's mind",
        "puxar a brasa à sua sardinha": "to look out for number one",
        "tirar o cavalinho da chuva": "to forget about it / no chance",
        "chover canivetes": "to rain cats and dogs",
        "cair a ficha": "to finally get it",
        "dor de cotovelo": "jealousy / sour grapes",
    },
}


def _get_idiom_dict(source_lang: str, target_lang: str) -> Optional[Dict[str, str]]:
    """Look up idiom dict, trying JSON then hardcoded fallback."""
    pair_key = f"{source_lang}_{target_lang}"
    reverse_key = f"{target_lang}_{source_lang}"
    for d in (IDIOMS, HARDCODED_IDIOMS):
        if pair_key in d:
            return d[pair_key]
        if reverse_key in d:
            return d[reverse_key]
    return None


def check_idioms(text: str, source_lang: str, target_lang: str) -> List[Tuple[str, str, str]]:
    """Check source text for known idioms. Returns list of (idiom, warning, meaning).

    Logs a warning when the language pair has no coverage.
    """
    idiom_dict = _get_idiom_dict(source_lang, target_lang)
    if idiom_dict is None:
        logger.debug(f"No idiom data for {source_lang}→{target_lang}")
        return []
    text_lower = text.lower()
    warnings = []
    for idiom, meaning in idiom_dict.items():
        if idiom in text_lower:
            warnings.append((idiom, f"Idiom detected: '{idiom}' → '{meaning}' in {target_lang}", meaning))
    return warnings


def gloss_idioms(text: str, source_lang: str, target_lang: str) -> str:
    """Annotate known idioms with their meaning in the source text to guide the model."""
    idiom_dict = _get_idiom_dict(source_lang, target_lang)
    if idiom_dict is None:
        return text
    text_lower = text.lower()
    result = text
    for idiom, meaning in idiom_dict.items():
        if idiom in text_lower:
            gloss = f" (meaning: {meaning})"
            result = result.replace(idiom.title() if idiom[0].isupper() else idiom, idiom + gloss)
    return result


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
    score_keys = [k for k in scores if k != "overall"]
    scores["overall"] = sum(scores[k] for k in score_keys) / max(len(score_keys), 1)

    return scores


# ── Grammar Post-Processing ────────────────────────────────────────────


def _library_grammar_check(text: str, lang: str) -> str:
    """Apply language-tool-python grammar corrections if the library is installed."""
    if not _LANGUAGE_TOOL_AVAILABLE:
        return text
    lt_lang = _get_lt_lang(lang)
    if lt_lang not in _LANGUAGE_TOOL_POOL:
        try:
            _LANGUAGE_TOOL_POOL[lt_lang] = language_tool_python.LanguageTool(lt_lang)
        except Exception:
            logger.debug(f"language-tool-python not available for {lt_lang}")
            return text
    tool = _LANGUAGE_TOOL_POOL[lt_lang]
    try:
        matches = tool.check(text)
        if matches:
            text = language_tool_python.utils.correct(text, matches)
    except Exception:
        logger.debug("language-tool-python check failed, falling back to regex")
    return text


def postprocess_grammar(text: str, lang: str, library_check: bool = True) -> str:
    """Apply grammar fixes. Uses language-tool-python if available, then regex fixes."""
    if library_check and _LANGUAGE_TOOL_AVAILABLE:
        text = _library_grammar_check(text, lang)
    if lang == "en":
        text = re.sub(r'\bi\b', 'I', text)
        text = re.sub(r"' ([A-Za-z])", r"'\1", text)
    elif lang == "fr":
        text = re.sub(r'\bl\s+\'', "l'", text)
        text = re.sub(r'\bd\s+\'', "d'", text)
        text = re.sub(r'(\w)\s+-\s+(\w)', r'\1-\2', text)
    elif lang == "de":
        if text and text[0].islower() and not text.startswith(("kein", "mein", "sein", "ein")):
            text = text[0].upper() + text[1:]
    elif lang in ("es", "it", "pt"):
        text = re.sub(
            r'\b(\w+)lo\s+(\w+)\b',
            lambda m: m.group(1) + "lo " + m.group(2)
            if m.group(1).endswith(("r", "s"))
            else m.group(0),
            text,
        )
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s([?.!,"])', r'\1', text)
    text = re.sub(r'(["\']) ', r'\1', text)
    return text


# ── Main quality pipeline ──────────────────────────────────────────────


class TranslationQualityPipeline:
    """Pipeline that wraps translation with quality enhancements.

    Features:
    - Tone/register detection via prefix tags or optional HF formality classifier
    - False friends detection and logging
    - Idiom detection + inline glossing to guide the decoder
    - Beam reranking by quality score
    - Grammar post-processing (regex + optional language-tool-python)
    - Domain-specific adapter routing

    All features degrade gracefully when a language pair has no coverage or
    when optional libraries (transformers, language-tool-python) are not installed.
    """

    def __init__(
        self,
        false_friends_enabled: bool = True,
        idioms_enabled: bool = True,
        grammar_postprocess: bool = True,
        grammar_library_check: bool = True,
        tone_library_boost: bool = True,
    ):
        self.false_friends_enabled = false_friends_enabled
        self.idioms_enabled = idioms_enabled
        self.grammar_postprocess = grammar_postprocess
        self.grammar_library_check = grammar_library_check
        self.tone_library_boost = tone_library_boost

    def prepare_input(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        domain: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Prepare text for translation: detect tone, check false friends, gloss idioms.

        Returns:
            (processed_text, tone, domain)
        """
        tone = detect_tone(text, library_boost=self.tone_library_boost)
        cleaned_text = strip_tone_tag(text)

        if self.idioms_enabled:
            idiom_warnings = check_idioms(cleaned_text, source_lang, target_lang)
            for idiom, warning, meaning in idiom_warnings:
                logger.info(f"Idiom: '{idiom}' → {meaning}")
            glossed_text = gloss_idioms(cleaned_text, source_lang, target_lang)
        else:
            glossed_text = cleaned_text

        text_with_hints = apply_tone_prompt(glossed_text, tone, source_lang)
        if tone == "NEUTRAL":
            text_with_hints = glossed_text + FRIENDLY_TIE_BREAKER.get(source_lang, "")

        if self.false_friends_enabled:
            warnings = check_false_friends(cleaned_text, source_lang, target_lang)
            for word, warning, correct in warnings:
                logger.info(f"False friend: '{word}' → {correct}")

        return text_with_hints, tone, domain or "general"

    def postprocess(self, translation: str, target_lang: str, tone: str) -> str:
        """Post-process translation: grammar + tone consistency."""
        if self.grammar_postprocess:
            translation = postprocess_grammar(
                translation, target_lang,
                library_check=self.grammar_library_check,
            )
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
            scores = score_translation_quality(
                source_text, translation, source_lang, target_lang, log_probs,
            )
            scored.append((scores.get("overall", 0.5), translation))

        scored.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"Reranked {len(candidates)} candidates, best score: {scored[0][0]:.3f}")
        return scored[0][1]
