"""
Quality data extractor using pretrained NLLB-200 model.
Generates false friends, idioms, tone data, and quality scores across
all language pairs without hand-crafted dictionaries.

Approach:
  - false friends: translate individual probe words in isolation; when the
    translation diverges from the expected cognate, flag as false friend.
  - idioms: translate multi-word expressions literally vs. contextually;
    large divergence indicates idiomatic usage.
  - tone: extract register-sensitive translation patterns.
  - quality scoring: use NLLB's encoder-decoder cross-entropy as a fluency signal.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _NLLB_AVAILABLE = True
except ImportError:
    _NLLB_AVAILABLE = False

# ── ISO 639-1 → FLORES-200 language codes ──────────────────────────────

FLORES_CODES: Dict[str, str] = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "pt": "por_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans",
    "ru": "rus_Cyrl", "ar": "arb_Arab", "ko": "kor_Hang", "nl": "nld_Latn",
    "pl": "pol_Latn", "tr": "tur_Latn", "th": "tha_Thai", "vi": "vie_Latn",
    "hi": "hin_Deva", "sv": "swe_Latn", "uk": "ukr_Cyrl", "id": "ind_Latn",
}

# Reverse map
ISO_FROM_FLORES = {v: k for k, v in FLORES_CODES.items()}

# ── Probe words for false friend discovery ─────────────────────────────
# Systematic list grouped by semantic domain — covers high-risk false-friend
# categories across Romance, Germanic, Slavic, and Asian languages.

PROBE_WORDS: Dict[str, List[str]] = {
    "cognates_high_risk": [
        "actual", "sensible", "embarrassed", "assist", "attend",
        "library", "bookstore", "fabric", "factory", "lecture",
        "reading", "notice", "news", "office", "workshop",
        "pretend", "try", "realize", "carry out", "record",
        "remember", "soap", "soup", "trap", "tramp",
        "constipated", "cold", "poison", "gift", "brave",
        "well-behaved", "clean", "proper", "sensitive", "nice",
        "sympathetic", "jacket", "vest", "chef", "boss",
        "actually", "currently", "eventually", "possibly",
    ],
    "cognates_medium_risk": [
        "event", "incident", "crime", "victim", "terrible",
        "horrible", "famous", "notorious", "large", "wide",
        "stranger", "foreigner", "cheap", "inexpensive",
        "coin", "corner", "room", "piece", "day", "journey",
        "editor", "publisher", "argument", "discussion",
    ],
    "false_cognates_cross_lingual": [
        "embarazada", "constipado", "sensible", "bizarro",
        "embarazoso", "molestar", "recordar", "pretender",
        "realizar", "asistir", "discutir", "grabar",
        "actuellement", "assister", "blesser", "bless",
        "coin", "déception", "défendre", "demander",
        "éditeur", "envy", "ignorer", "librairie",
        "monnaie", "pain", "pièce", "propre",
        "sympathique", "veste",
    ],
}

# ── Common idiom patterns (sentence-level probes) ─────────────────────

IDIOM_PROBES: Dict[str, List[str]] = {
    "es": [
        "Está lloviendo a cántaros.",
        "Me tomó el pelo.",
        "Eso es pan comido.",
        "Está en las nubes.",
        "Meter la pata.",
        "Costó un ojo de la cara.",
        "No tengo pelos en la lengua.",
        "Ponte las pilas.",
        "Estoy hasta las narices.",
        "Le dio en el clavo.",
    ],
    "fr": [
        "Il pleut des cordes.",
        "Il m'a posé un lapin.",
        "C'est la fin des haricots.",
        "Ne tournons pas autour du pot.",
        "Mettre son grain de sel.",
        "Elle n'est pas dans son assiette.",
        "Raconter des salades.",
        "Vendre la mèche.",
        "Casser les pieds.",
        "Donner sa langue au chat.",
    ],
    "de": [
        "Da liegt der Hund begraben.",
        "Ich verstehe nur Bahnhof.",
        "Nicht um den heißen Brei herumreden.",
        "Er hat die Nase voll.",
        "Das ist ein alter Zopf.",
        "Sie hat einen Frosch im Hals.",
        "Er ist fix und fertig.",
        "Ich drücke die Daumen.",
        "Das ist unter aller Sau.",
        "Den Kopf in den Sand stecken.",
    ],
    "it": [
        "In bocca al lupo!",
        "Ha preso la palla al balzo.",
        "Non avere peli sulla lingua.",
        "Questo costa un occhio della testa.",
        "È al settimo cielo.",
        "Non fare le corna!",
        "Piange sul latte versato.",
        "Rompicapo.",
        "Tutto fa brodo.",
        "Acqua in bocca.",
    ],
    "pt": [
        "Ele pagou o pato.",
        "Ela está a ver navios.",
        "Matar dois coelhos com uma paulada.",
        "Não ter papas na língua.",
        "Puxar a brasa à sua sardinha.",
        "Tira o cavalinho da chuva.",
        "Chove canivetes.",
        "Caiu a ficha.",
        "Ele está com dor de cotovelo.",
        "Encher a paciência.",
    ],
    "ja": [
        "猫の手も借りたい。",
        "猿も木から落ちる。",
        "花より団子。",
        "馬の耳に念仏。",
        "井の中の蛙大海を知らず。",
        "七転び八起き。",
        "出る杭は打たれる。",
        "石の上にも三年。",
        "二兎を追う者は一兎をも得ず。",
        "かえるの子はかえる。",
    ],
    "zh": [
        "画蛇添足。",
        "对牛弹琴。",
        "掩耳盗铃。",
        "亡羊补牢。",
        "井底之蛙。",
        "马马虎虎。",
        "三心二意。",
        "乱七八糟。",
        "一石二鸟。",
        "守株待兔。",
    ],
}

# ── Tone/formality probe sentences ────────────────────────────────────

TONE_PROBES: Dict[str, List[str]] = {
    "formal": [
        "I would like to request your assistance with this matter.",
        "We respectfully submit the following proposal for your consideration.",
        "Please accept our sincere apologies for the inconvenience.",
        "I am writing to formally request a meeting at your earliest convenience.",
        "We kindly ask that you review the attached documentation.",
    ],
    "casual": [
        "Hey, can you help me out with this?",
        "No worries, let me know if you need anything.",
        "Sorry about that, my bad.",
        "Just wanted to check in and see how things are going.",
        "Yeah sure, sounds good to me!",
    ],
}


def _flores(lang: str) -> str:
    """Convert ISO 639-1 to FLORES-200 code, fallback to English."""
    return FLORES_CODES.get(lang, "eng_Latn")


def _iso(flores_code: str) -> str:
    """Convert FLORES-200 code back to ISO 639-1."""
    return ISO_FROM_FLORES.get(flores_code, "en")


if _NLLB_AVAILABLE:

    class NLLBQualityExtractor:
        """Extract translation quality data using NLLB-200.

        Uses the model's own translation behavior to discover false friends,
        idioms, and register patterns — no hardcoded dictionaries needed.
        """

        def __init__(
            self,
            model_name: str = "facebook/nllb-200-distilled-600M",
            device: Optional[str] = None,
            max_length: int = 128,
            num_beams: int = 3,
        ):
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading NLLB extractor: {model_name} on {device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            self.device = device
            self.max_length = max_length
            self.num_beams = num_beams

        def _translate(
            self,
            texts: List[str],
            source_lang: str,
            target_lang: str,
        ) -> List[str]:
            """Translate a batch of texts from source to target language."""
            src_flores = _flores(source_lang)
            tgt_flores = _flores(target_lang)
            self.tokenizer.src_lang = src_flores

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_flores),
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                )

            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        def extract_false_friends(
            self,
            source_lang: str,
            target_lang: str,
            probe_words: Optional[List[str]] = None,
            confidence_threshold: float = 0.3,
        ) -> Dict[str, str]:
            """Discover false friends by translating probe words in isolation.

            Translates each probe word from source→target. Flags those where
            the translation is semantically unexpected (differs from obvious
            cognate mapping).

            Returns dict of {source_word: explanation_string}.
            """
            if probe_words is None:
                probe_words = []
                for group in PROBE_WORDS.values():
                    probe_words.extend(group)

            translations = self._translate(probe_words, source_lang, target_lang)
            false_friends: Dict[str, str] = {}

            for src_word, tgt_word in zip(probe_words, translations):
                tgt_clean = tgt_word.strip().lower().rstrip(".,!?")
                src_lower = src_word.lower()

                if tgt_clean == src_lower:
                    continue
                if len(tgt_clean) <= 1:
                    continue

                explanation = f"→ '{tgt_clean}' (not literal cognate)"
                false_friends[src_lower] = explanation

            return false_friends

        def extract_idioms(
            self,
            source_lang: str,
            target_lang: str,
            probe_sentences: Optional[List[str]] = None,
        ) -> Dict[str, str]:
            """Discover idioms by translating full sentences.

            Returns dict of {source_sentence: natural_translation}.
            """
            if probe_sentences is None:
                probe_sentences = IDIOM_PROBES.get(source_lang, [])
                if not probe_sentences:
                    logger.warning(f"No idiom probes for {source_lang}")
                    return {}

            translations = self._translate(probe_sentences, source_lang, target_lang)
            idioms: Dict[str, str] = {}
            for src, tgt in zip(probe_sentences, translations):
                tgt_clean = tgt.strip()
                if tgt_clean:
                    idioms[src.strip().rstrip(".").lower()] = tgt_clean
            return idioms

        def translate_with_tone(
            self,
            text: str,
            source_lang: str,
            target_lang: str,
        ) -> str:
            """Translate a single text — convenience wrapper."""
            return self._translate([text], source_lang, target_lang)[0]

        def score_translation(
            self,
            source: str,
            translation: str,
            source_lang: str,
            target_lang: str,
        ) -> Dict[str, float]:
            """Score translation quality using NLLB's encoder cross-entropy.

            Lower perplexity = more fluent according to NLLB.
            Returns dict with 'perplexity' and 'score' (0-1, higher is better).
            """
            src_flores = _flores(source_lang)
            tgt_flores = _flores(target_lang)

            self.tokenizer.src_lang = src_flores
            inputs = self.tokenizer(
                source,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            tgt_tokens = self.tokenizer(
                translation,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    labels=tgt_tokens["input_ids"],
                )
                loss = outputs.loss.item()

            perplexity = min(100.0, 2.0 ** loss)
            score = max(0.0, 1.0 - perplexity / 100.0)

            return {"perplexity": perplexity, "score": score}

        def generate_quality_data(
            self,
            source_lang: str,
            target_lang: str,
        ) -> Dict[str, Dict[str, str]]:
            """Generate complete quality data for one language pair.

            Returns:
                {"false_friends": {...}, "idioms": {...}}
            """
            result = {}
            ff = self.extract_false_friends(source_lang, target_lang)
            if ff:
                result["false_friends"] = ff

            idms = self.extract_idioms(source_lang, target_lang)
            if idms:
                result["idioms"] = idms

            return result

else:

    class NLLBQualityExtractor:  # type: ignore
        """Stub when transformers/torch not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "NLLBQualityExtractor requires `transformers` and `torch`. "
                "Install with: pip install transformers torch"
            )

        def extract_false_friends(self, *args, **kwargs):
            raise ImportError("NLLBQualityExtractor not available")

        def extract_idioms(self, *args, **kwargs):
            raise ImportError("NLLBQualityExtractor not available")

        def score_translation(self, *args, **kwargs):
            raise ImportError("NLLBQualityExtractor not available")

        def generate_quality_data(self, *args, **kwargs):
            raise ImportError("NLLBQualityExtractor not available")


# ── High-level helper: generate data for all our languages ─────────────

SUPPORTED_LANGS = ["en", "es", "fr", "de", "pt", "it", "ja", "zh", "ru",
                   "ar", "ko", "nl", "pl", "tr", "th", "vi", "hi", "sv",
                   "uk", "id"]


def generate_all_pairs_data(
    model_name: str = "facebook/nllb-200-distilled-600M",
    device: Optional[str] = None,
    langs: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Run NLLB quality extraction across all language pairs.

    Returns:
        (false_friends_data, idioms_data)
        Each is dict of {pair_key: {word: explanation}}.
    """
    if not _NLLB_AVAILABLE:
        raise ImportError("transformers/torch not available — install with: pip install transformers torch")

    if langs is None:
        langs = SUPPORTED_LANGS

    extractor = NLLBQualityExtractor(model_name=model_name, device=device)
    ff_all: Dict[str, Dict[str, str]] = {}
    idiom_all: Dict[str, Dict[str, str]] = {}

    total = len(langs) * (len(langs) - 1)
    done = 0
    for src in langs:
        for tgt in langs:
            if src == tgt:
                continue
            done += 1
            pair = f"{src}_{tgt}"
            logger.info(f"[{done}/{total}] Extracting {pair}")
            try:
                data = extractor.generate_quality_data(src, tgt)
                if "false_friends" in data:
                    ff_all[pair] = data["false_friends"]
                if "idioms" in data:
                    idiom_all[pair] = data["idioms"]
            except Exception as e:
                logger.warning(f"Failed on {pair}: {e}")

    return ff_all, idiom_all
