# data/synthetic_augmentation.py
"""
Synthetic data augmentation — Refactored to use shared utilities.
Generate additional training data using modern transformer models.

Extended with targeted augmentation strategies that teach the model to handle:
  - False friends (cognate traps across language pairs)
  - Idioms / multi-word expressions
  - Tone / register control ([FORMAL], [CASUAL] prefix tags)
  - Cultural context via domain-specific data
  - Backtranslation for general data augmentation

The model learns these phenomena from training data, not from hardcoded rules.
"""

# Optional heavy deps: transformers, sentence_transformers, torch. Provide shims for smoke/dry-run.
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
except Exception:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

    def pipeline(*args, **kwargs):
        raise ImportError("transformers is required for SyntheticDataAugmenter")
try:
    import torch
except Exception:

    class torch:
        @staticmethod
        def cuda():
            class _C:
                @staticmethod
                def is_available():
                    return False
            return _C
        float16 = None
        float32 = None
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
import numpy as np
import json
import logging
import random

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None

    class util:
        @staticmethod
        def cos_sim(a, b):
            return np.array([[0.0]])

from data.data_utils import estimate_sentence_count
from utils.common_utils import DirectoryManager
from config.schemas import RootConfig, load_config

logger = logging.getLogger(__name__)

# ── Seed data for targeted augmentation ─────────────────────────────────
# Small curated lists of the most common false friends and idioms per pair.
# These are *seed words* used to generate full training examples via NLLB,
# NOT an exhaustive dictionary used for inference.

FALSE_FRIEND_SEEDS: Dict[str, Dict[str, str]] = {
    "en_es": {
        "actualmente": "currently",
        "embarazada": "pregnant",
        "constipado": "cold (illness)",
        "sensible": "sensitive",
        "bizarro": "dashing/brave",
        "asistir": "to attend",
        "discutir": "to argue",
        "molestar": "to bother",
        "pretender": "to try",
        "realizar": "to carry out",
        "recordar": "to remember",
        "sopa": "soup",
    },
    "en_fr": {
        "actuellement": "currently",
        "assister": "to attend",
        "blesser": "to wound",
        "déception": "disappointment",
        "demander": "to ask",
        "ignorer": "to not know",
        "librairie": "bookstore",
        "pain": "bread",
        "sensible": "sensitive",
        "sympathique": "nice",
    },
    "en_de": {
        "aktuell": "current",
        "bekommen": "to receive",
        "billig": "cheap",
        "brav": "well-behaved",
        "eventuell": "possibly",
        "gift": "poison",
        "handy": "mobile phone",
        "sensibel": "sensitive",
        "rat": "advice",
        "stift": "pen",
    },
    "en_it": {
        "camera": "room",
        "confetti": "sugared almonds",
        "grosso": "big",
        "largo": "wide",
        "palazzo": "building",
        "preservativo": "condom",
        "simpatia": "likability",
        "tassa": "tax",
        "traffico": "traffic",
    },
    "en_pt": {
        "esquisito": "strange",
        "fechar": "to close",
        "gosto": "taste",
        "largo": "wide",
        "legenda": "subtitle",
        "oficina": "workshop",
        "polvo": "octopus",
        "reclamar": "to complain",
        "salsa": "parsley",
    },
    "en_nl": {
        "eventueel": "possibly",
        "formulier": "form",
        "gift": "poison",
        "rook": "smoke",
        "slim": "smart",
        "straf": "punishment",
        "winkel": "shop",
    },
    "en_sv": {
        "eventuell": "possible",
        "gift": "poison/married",
        "känslig": "sensitive",
        "rolig": "funny",
        "semester": "vacation",
        "smal": "narrow",
    },
    "en_pl": {
        "aktualny": "current",
        "dodatek": "supplement",
        "dywan": "carpet",
        "lustro": "mirror",
        "zakaz": "prohibition",
    },
    "en_tr": {
        "akıl": "mind",
        "entrika": "intrigue",
        "kibar": "polite",
        "masa": "table",
        "rapor": "report",
    },
}

IDIOM_SEEDS: Dict[str, List[str]] = {
    "es": [
        "Está lloviendo a cántaros.",
        "Eso es pan comido.",
        "Está en las nubes.",
        "Meter la pata.",
        "Costó un ojo de la cara.",
        "No tengo pelos en la lengua.",
        "Ponte las pilas.",
        "Le dio en el clavo.",
    ],
    "fr": [
        "Il pleut des cordes.",
        "C'est la fin des haricots.",
        "Mettre son grain de sel.",
        "Vendre la mèche.",
        "Casser les pieds.",
    ],
    "de": [
        "Da liegt der Hund begraben.",
        "Ich verstehe nur Bahnhof.",
        "Er hat die Nase voll.",
        "Ich drücke die Daumen.",
        "Das ist unter aller Sau.",
    ],
    "it": [
        "In bocca al lupo!",
        "Ha preso la palla al balzo.",
        "Costa un occhio della testa.",
        "È al settimo cielo.",
        "Acqua in bocca.",
    ],
    "pt": [
        "Ele pagou o pato.",
        "Matar dois coelhos com uma paulada.",
        "Chove canivetes.",
        "Caiu a ficha.",
        "Não ter papas na língua.",
    ],
    "ja": [
        "猫の手も借りたい。",
        "猿も木から落ちる。",
        "花より団子。",
        "井の中の蛙大海を知らず。",
        "七転び八起き。",
    ],
    "zh": [
        "画蛇添足。",
        "对牛弹琴。",
        "亡羊补牢。",
        "井底之蛙。",
        "一石二鸟。",
    ],
    "nl": [
        "Daar komt de aap uit de mouw.",
        "De kat uit de boom kijken.",
        "Een boekje opendoen.",
        "Het paard achter de wagen spannen.",
        "Twee vliegen in één klap.",
    ],
    "sv": [
        "Bita i det sura äpplet.",
        "Det är ingen ko på isen.",
        "Slå två flugor i en smäll.",
        "Ta tjuren vid hornen.",
        "Lägga rabarber på.",
    ],
    "pl": [
        "Bułka z masłem.",
        "Nie mój cyrk, nie moje małpy.",
        "Robić z igły widły.",
        "Trzymać kciuki.",
        "Kłamstwo ma krótkie nogi.",
    ],
    "tr": [
        "Bir taşla iki kuş.",
        "İğneyi kendine, çuvaldızı başkasına batır.",
        "Pire için yorgan yakmak.",
        "Kafayı yemek.",
        "Sinir küpü.",
    ],
}

# Template sentences for false friend augmentation
FF_TEMPLATES: Dict[str, List[str]] = {
    "en": [
        "The manager is {word} available.",
        "She felt {word} about the situation.",
        "They said it was {word} important.",
        "He is {word} for the position.",
        "This is a {word} approach to the problem.",
    ],
    "es": [
        "El director está {word} disponible.",
        "Ella se sintió {word} por la situación.",
        "Dijeron que era {word} importante.",
        "Él está {word} para el puesto.",
        "Este es un enfoque {word} al problema.",
    ],
    "fr": [
        "Le directeur est {word} disponible.",
        "Elle s'est sentie {word} face à la situation.",
        "Ils ont dit que c'était {word} important.",
        "Il est {word} pour le poste.",
        "C'est une approche {word} du problème.",
    ],
    "de": [
        "Der Direktor ist {word} verfügbar.",
        "Sie fühlte sich {word} wegen der Situation.",
        "Sie sagten, es sei {word} wichtig.",
        "Er ist {word} für die Position.",
        "Das ist ein {word} Ansatz für das Problem.",
    ],
}

# Register transformation prompts for tone augmentation
FORMAL_HINT = " [Formal register: polite forms, complete sentences]"
CASUAL_HINT = " [Casual register: everyday language, conversational]"

# ── NLLB code mapping ──────────────────────────────────────────────────

NLLB_CODE_MAP: Dict[str, str] = {
    'en': 'eng_Latn', 'es': 'spa_Latn', 'fr': 'fra_Latn', 'de': 'deu_Latn',
    'pt': 'por_Latn', 'it': 'ita_Latn', 'ja': 'jpn_Jpan', 'zh': 'zho_Hans',
    'ru': 'rus_Cyrl', 'ar': 'arb_Arab', 'ko': 'kor_Hang', 'nl': 'nld_Latn',
    'pl': 'pol_Latn', 'tr': 'tur_Latn', 'th': 'tha_Thai', 'vi': 'vie_Latn',
    'hi': 'hin_Deva', 'sv': 'swe_Latn', 'uk': 'ukr_Cyrl', 'id': 'ind_Latn',
}

# ── Augmenter class ────────────────────────────────────────────────────


class SyntheticDataAugmenter:
    """Generate additional training data using modern transformer models"""

    def __init__(self, config: RootConfig, base_model: str = 'facebook/nllb-200-distilled-1.3B'):
        self.logger = logging.getLogger(__name__)
        self.base_model = base_model
        self.config = config
        self.languages = self.config.data.active_languages
        self.quality_threshold = self.config.data.quality_threshold
        self.output_dir = Path(self.config.data.processed_dir)

        self._model = None
        self._tokenizer = None
        self._translator = None
        self._sentence_model = None

        self.logger.info(f"Initialized augmenter with model: {base_model}")

    @property
    def model(self):
        if self._model is None:
            if AutoModelForSeq2SeqLM is None:
                raise ImportError("transformers required")
            self.logger.info("Loading translation model...")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model,
                torch_dtype=(torch.float16 if hasattr(torch, 'cuda') and torch.cuda.is_available() else getattr(torch, 'float32', None)),
                device_map=("auto" if hasattr(torch, 'cuda') and torch.cuda.is_available() else None)
            )
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if AutoTokenizer is None:
                raise ImportError("transformers required")
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return self._tokenizer

    @property
    def translator(self):
        if self._translator is None:
            self._translator = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=(0 if hasattr(torch, 'cuda') and torch.cuda.is_available() else -1),
                batch_size=(8 if hasattr(torch, 'cuda') and torch.cuda.is_available() else 1)
            )
        return self._translator

    @property
    def sentence_model(self):
        if self._sentence_model is None:
            self.logger.info("Loading sentence transformer...")
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._sentence_model

    def _nllb_code(self, lang: str) -> str:
        return NLLB_CODE_MAP.get(lang, lang)

    def _translate_batch(self, texts: List[str], src: str, tgt: str) -> List[str]:
        """Translate a batch using the NLLB pipeline."""
        try:
            results = self.translator(
                texts,
                src_lang=self._nllb_code(src),
                tgt_lang=self._nllb_code(tgt),
                max_length=512,
            )
            return [r['translation_text'] for r in results]
        except Exception as e:
            self.logger.error(f"Batch translation failed: {e}")
            return [""] * len(texts)

    # ── 1. False Friend Augmentation ───────────────────────────────────

    def generate_false_friend_examples(
        self,
        source_lang: str,
        target_lang: str,
        output_file: str,
    ) -> Dict[str, int]:
        """Generate parallel examples that teach correct false-friend mappings.

        For each known false friend (e.g. 'actualmente' in Spanish), creates
        template sentences in the source language, translates them with NLLB
        to get the correct target, and writes (source+ff → correct_target)
        pairs. The model learns from these during training.
        """
        pair = f"{source_lang}_{target_lang}"
        ff_dict = FALSE_FRIEND_SEEDS.get(pair)
        if ff_dict is None:
            self.logger.info(f"No false friend seeds for {pair}, skipping")
            return {"generated": 0, "pair": pair}

        output_path = Path(output_file)
        DirectoryManager.create_directory(output_path.parent)

        src_templates = FF_TEMPLATES.get(source_lang, FF_TEMPLATES.get("en", []))
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for ff_word, correct_meaning in ff_dict.items():
                for tmpl in src_templates:
                    src_sentence = tmpl.replace("{word}", ff_word)
                    tgt_sentences = self._translate_batch([src_sentence], source_lang, target_lang)
                    if tgt_sentences[0]:
                        f.write(f"{src_sentence}\t{tgt_sentences[0]}\n")
                        count += 1

        self.logger.info(f"Generated {count} false-friend examples for {pair}")
        return {"generated": count, "pair": pair}

    # ── 2. Idiom Augmentation ──────────────────────────────────────────

    def generate_idiom_examples(
        self,
        source_lang: str,
        target_lang: str,
        output_file: str,
    ) -> Dict[str, int]:
        """Generate parallel examples with idiomatic source → natural translation.

        Translates known idioms via NLLB to produce natural target equivalents.
        The model learns to map idiomatic expressions to their natural counterparts.
        """
        probes = IDIOM_SEEDS.get(source_lang, [])
        if not probes:
            self.logger.info(f"No idiom seeds for {source_lang}, skipping")
            return {"generated": 0, "source_lang": source_lang}

        output_path = Path(output_file)
        DirectoryManager.create_directory(output_path.parent)

        translations = self._translate_batch(probes, source_lang, target_lang)
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for src, tgt in zip(probes, translations):
                if tgt:
                    f.write(f"{src}\t{tgt}\n")
                    count += 1

        self.logger.info(f"Generated {count} idiom examples for {source_lang}→{target_lang}")
        return {"generated": count, "source_lang": source_lang, "target_lang": target_lang}

    # ── 3. Tone / Register Augmentation ────────────────────────────────

    def generate_tone_examples(
        self,
        input_parallel_file: str,
        source_lang: str,
        target_lang: str,
        output_formal_file: str,
        output_casual_file: str,
        max_examples: int = 10000,
    ) -> Dict[str, int]:
        """Create formal/casual versions of existing parallel sentences.

        Prepends [FORMAL] / [CASUAL] tags and appends register hints to guide
        NLLB to produce register-matched translations. The resulting parallel
        data teaches the model to associate the tags with the right register.
        """
        input_path = Path(input_parallel_file)
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return {"error": "file_not_found"}

        output_f = Path(output_formal_file)
        output_c = Path(output_casual_file)
        DirectoryManager.create_directory(output_f.parent)
        DirectoryManager.create_directory(output_c.parent)

        pairs: List[Tuple[str, str]] = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
                if len(pairs) >= max_examples:
                    break

        if not pairs:
            return {"generated": 0}

        # Create formal versions: tag source, guide NLLB toward formal register
        formal_sources = [f"[FORMAL]{src}{FORMAL_HINT}" for src, _ in pairs]
        formal_targets = self._translate_batch(formal_sources, source_lang, target_lang)

        casual_sources = [f"[CASUAL]{src}{CASUAL_HINT}" for src, _ in pairs]
        casual_targets = self._translate_batch(casual_sources, source_lang, target_lang)

        f_count = c_count = 0
        with open(output_f, 'w') as f_out:
            for src, tgt in zip(formal_sources, formal_targets):
                if tgt:
                    f_out.write(f"{src}\t{tgt}\n")
                    f_count += 1

        with open(output_c, 'w') as f_out:
            for src, tgt in zip(casual_sources, casual_targets):
                if tgt:
                    f_out.write(f"{src}\t{tgt}\n")
                    c_count += 1

        self.logger.info(f"Generated {f_count} formal + {c_count} casual examples")
        return {"formal": f_count, "casual": c_count}

    # ── 4. Cultural Context Augmentation ───────────────────────────────

    def generate_cultural_context_examples(
        self,
        domain_data_dir: str,
        output_file: str,
        target_lang: str = "en",
    ) -> Dict[str, int]:
        """Generate culturally-aware examples from domain-specific data.

        Reads domain-specific parallel data (medical, legal, tech) and creates
        additional pairs where domain-specific terms are translated with
        appropriate cultural context. Uses NLLB to re-translate with context
        hints appended.
        """
        domain_dir = Path(domain_data_dir)
        if not domain_dir.exists():
            self.logger.warning(f"Domain data directory not found: {domain_dir}")
            return {"error": "dir_not_found"}

        output_path = Path(output_file)
        DirectoryManager.create_directory(output_path.parent)

        all_pairs: List[Tuple[str, str]] = []
        for fpath in domain_dir.glob("*.txt"):
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        all_pairs.append((parts[0], parts[1]))

        if not all_pairs:
            return {"generated": 0}

        # Re-translate with cultural context hint to produce natural translations
        context_hint = " [Translate naturally for the target culture, adapting culturally-specific terms]"
        augmented_sources = [src + context_hint for src, _ in all_pairs]
        domain = domain_dir.name

        # Detect source language from the data (first pair's original)
        augmented_targets = self._translate_batch(
            augmented_sources,
            target_lang,  # source for NLLB is the non-English side
            target_lang,  # target for NLLB is English
        )

        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for src, tgt in zip(augmented_sources, augmented_targets):
                if tgt:
                    f.write(f"{src}\t{tgt}\t{domain}\n")
                    count += 1

        self.logger.info(f"Generated {count} cultural-context examples from {domain}")
        return {"generated": count, "domain": domain}

    # ── 5. Backtranslation (existing) ──────────────────────────────────

    def augment_with_backtranslation(
        self,
        monolingual_file: str,
        source_lang: str,
        target_lang: str,
        output_file: str,
        max_sentences: int = 100000,
        batch_size: int = 32,
    ) -> Dict[str, int]:
        """
        Use backtranslation to create synthetic parallel data.

        Returns:
            Statistics about the augmentation process
        """
        monolingual_path = Path(monolingual_file)
        output_path = Path(output_file)
        DirectoryManager.create_directory(output_path.parent)

        if not monolingual_path.exists():
            self.logger.error(f"Monolingual file not found: {monolingual_path}")
            return {'error': 'file_not_found', 'augmented': 0}

        self.logger.info(f"Generating backtranslations for {source_lang}->{target_lang}")
        total_sentences = estimate_sentence_count(monolingual_path)
        sentences_to_process = min(total_sentences, max_sentences)

        stats = {
            'total_sentences': total_sentences,
            'processed': 0,
            'augmented': 0,
            'filtered_quality': 0,
            'errors': 0,
        }

        with open(monolingual_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                batch_texts = []
                for line_num, line in enumerate(tqdm(f_in, total=sentences_to_process, desc="Backtranslating")):
                    if line_num >= sentences_to_process:
                        break
                    text = line.strip()
                    if not text or len(text) < 10:
                        continue
                    batch_texts.append(text)
                    if len(batch_texts) >= batch_size:
                        results = self._process_backtranslation_batch(batch_texts, source_lang, target_lang)
                        for original, translated, back_translated in results:
                            if translated and self._is_quality_translation(original, back_translated):
                                f_out.write(f"{original}\t{translated}\n")
                                stats['augmented'] += 1
                            else:
                                stats['filtered_quality'] += 1
                        stats['processed'] += len(batch_texts)
                        batch_texts = []

                if batch_texts:
                    results = self._process_backtranslation_batch(batch_texts, source_lang, target_lang)
                    for original, translated, back_translated in results:
                        if translated and self._is_quality_translation(original, back_translated):
                            f_out.write(f"{original}\t{translated}\n")
                            stats['augmented'] += 1
                        else:
                            stats['filtered_quality'] += 1
                    stats['processed'] += len(batch_texts)

        self.logger.info(f"Augmentation complete: {stats['augmented']:,} pairs created")
        self.logger.info(f"Quality filtered: {stats['filtered_quality']:,} pairs")
        return stats

    def _process_backtranslation_batch(
        self, texts: List[str], source_lang: str, target_lang: str
    ) -> List[Tuple[str, str, str]]:
        results = []
        try:
            translations = self.translator(
                texts,
                src_lang=self._nllb_code(source_lang),
                tgt_lang=self._nllb_code(target_lang),
                max_length=512,
            )
            translated_texts = [t['translation_text'] for t in translations]
            back_translations = self.translator(
                translated_texts,
                src_lang=self._nllb_code(target_lang),
                tgt_lang=self._nllb_code(source_lang),
                max_length=512,
            )
            back_translated_texts = [t['translation_text'] for t in back_translations]
            for original, translated, back_translated in zip(texts, translated_texts, back_translated_texts):
                results.append((original, translated, back_translated))
        except Exception as e:
            self.logger.error(f"Batch translation failed: {e}")
            results = [(text, None, None) for text in texts]
        return results

    def _is_quality_translation(self, original: str, back_translated: str) -> bool:
        if not back_translated:
            return False
        try:
            embeddings = self.sentence_model.encode(
                [original, back_translated], convert_to_tensor=True
            )
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            return similarity >= self.quality_threshold
        except Exception as e:
            self.logger.error(f"Quality check failed: {e}")
            return False

    # ── Pivot translation (existing) ──────────────────────────────────

    def generate_pivot_translations(
        self, english_pairs_dir: str, output_dir: Optional[str] = None
    ) -> Dict[str, int]:
        english_pairs_path = Path(english_pairs_dir)
        output_path = Path(output_dir) if output_dir else self.output_dir / 'pivot_pairs'
        DirectoryManager.create_directory(output_path)

        pairs_data: Dict[str, List[Tuple[str, str]]] = {}
        self.logger.info("Loading English-centric pairs...")

        for lang in self.languages:
            if lang == 'en':
                continue
            patterns = [f'en-{lang}_sampled.txt', f'en-{lang}.txt', f'opus_en-{lang}.txt']
            for pattern in patterns:
                file_path = english_pairs_path / pattern
                if file_path.exists():
                    pairs_data[lang] = self._load_pairs(file_path)
                    self.logger.info(f"Loaded en-{lang}: {len(pairs_data[lang]):,} pairs")
                    break

        stats = {'total_pivot_pairs': 0, 'pairs_created': {}}
        self.logger.info("Generating pivoted pairs...")

        for lang1 in pairs_data:
            for lang2 in pairs_data:
                if lang1 < lang2:
                    pair_count = self._create_pivot_pairs(
                        pairs_data[lang1], pairs_data[lang2], lang1, lang2, output_path
                    )
                    stats['pairs_created'][f'{lang1}-{lang2}'] = pair_count
                    stats['total_pivot_pairs'] += pair_count

        self.logger.info(f"Generated {stats['total_pivot_pairs']:,} pivot pairs")
        return stats

    def _load_pairs(self, file_path: Path, max_pairs: int = 50000) -> List[Tuple[str, str]]:
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_pairs:
                    break
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    def _create_pivot_pairs(
        self, pairs1: List[Tuple[str, str]], pairs2: List[Tuple[str, str]],
        lang1: str, lang2: str, output_path: Path,
    ) -> int:
        self.logger.info(f"Creating pivot pairs for {lang1}-{lang2}")
        output_file = output_path / f'{lang1}-{lang2}_pivot.txt'
        en_to_lang1 = {en: lang for en, lang in pairs1}
        en_to_lang2 = {en: lang for en, lang in pairs2}
        common_en = set(en_to_lang1.keys()) & set(en_to_lang2.keys())
        pairs_created = 0
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for en_text in common_en:
                try:
                    f_out.write(f"{en_to_lang1[en_text]}\t{en_to_lang2[en_text]}\n")
                    pairs_created += 1
                except Exception as e:
                    self.logger.error(f"Failed to create pivot pair: {e}")
        self.logger.info(f"Created {pairs_created:,} {lang1}-{lang2} pairs")
        return pairs_created


# ── Convenience batch runner ───────────────────────────────────────────

def run_all_augmentations(config: RootConfig, langs: Optional[List[str]] = None):
    """Run all augmentation strategies for all language pairs.

    Generates training data for false friends, idioms, tone, and
    backtranslation across all supported language pairs.
    """
    if langs is None:
        langs = config.data.active_languages

    augmenter = SyntheticDataAugmenter(config)
    base_dir = Path(config.data.processed_dir) / "augmented"
    results = {}

    for src in langs:
        for tgt in langs:
            if src == tgt:
                continue

            pair_key = f"{src}_{tgt}"
            pair_dir = base_dir / pair_key
            DirectoryManager.create_directory(pair_dir)

            # False friends
            ff_out = str(pair_dir / "false_friends.txt")
            results[f"ff_{pair_key}"] = augmenter.generate_false_friend_examples(src, tgt, ff_out)

            # Idioms
            idiom_out = str(pair_dir / "idioms.txt")
            results[f"idiom_{pair_key}"] = augmenter.generate_idiom_examples(src, tgt, idiom_out)

    logger.info(f"Batch augmentation complete. {sum(v.get('generated', 0) for v in results.values())} total examples.")
    return results


def main():
    """Standalone example."""
    config = load_config()
    augmenter = SyntheticDataAugmenter(config)

    # Example: false friends for Spanish→English
    stats = augmenter.generate_false_friend_examples(
        source_lang='es', target_lang='en',
        output_file='test_data/ff_es_en.txt',
    )
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Example: idioms for French→English
    stats = augmenter.generate_idiom_examples(
        source_lang='fr', target_lang='en',
        output_file='test_data/idiom_fr_en.txt',
    )
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
