"""
Wikipedia → Backtranslation pipeline.

Downloads Wikipedia dumps for all supported languages, cleans them,
writes monolingual corpora, then translates them via NLLB to/from English
to produce synthetic parallel pairs.
"""
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

WIKIPEDIA_LANG_CODES = {
    'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'pt': 'pt',
    'it': 'it', 'ja': 'ja', 'zh': 'zh', 'ru': 'ru', 'ar': 'ar',
    'ko': 'ko', 'nl': 'nl', 'pl': 'pl', 'tr': 'tr', 'th': 'th',
    'vi': 'vi', 'hi': 'hi', 'sv': 'sv', 'uk': 'uk', 'id': 'id',
}

WIKIPEDIA_DATE = "20220301"


def _strip_wiki_markup(text: str) -> str:
    """Remove common Wikipedia markup artifacts."""
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\[\[[^]]*\|', '', text)
    text = re.sub(r'[\[\]\']', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _clean_sentence(text: str) -> Optional[str]:
    """Clean and filter a single sentence."""
    text = _strip_wiki_markup(text)
    if not text or len(text) < 20 or len(text) > 512:
        return None
    if sum(1 for c in text if c.isalpha()) < len(text) * 0.3:
        return None
    return text


def download_wikipedia_corpus(
    lang: str,
    max_sentences: int = 500_000,
    date: str = WIKIPEDIA_DATE,
) -> List[str]:
    """Download Wikipedia dump for a language, clean, and deduplicate."""
    if load_dataset is None:
        logger.error("datasets library not installed")
        return []

    dataset_name = f"wikipedia"
    config_name = f"{date}.{lang}"

    logger.info(f"Downloading Wikipedia {lang}...")
    try:
        dataset = load_dataset(
            dataset_name,
            config_name,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(f"Wikipedia download failed for {lang}: {e}")
        try:
            dataset = load_dataset(
                dataset_name,
                config_name,
                split="train",
                streaming=True,
                trust_remote_code=False,
            )
        except Exception as e2:
            logger.error(f"Wikipedia download also failed without trust_remote_code: {e2}")
            return []

    sentences: Set[str] = set()
    seen = 0
    for example in dataset:
        text = example.get("text", "")
        cleaned = _clean_sentence(text)
        if cleaned and cleaned not in sentences:
            sentences.add(cleaned)
        seen += 1
        if len(sentences) >= max_sentences:
            break
        if seen % 50_000 == 0:
            logger.info(f"  {lang}: scanned {seen:,}, kept {len(sentences):,}")

    logger.info(f"Wikipedia {lang}: {len(sentences):,} unique sentences from {seen:,} scanned")
    return list(sentences)


class WikipediaBacktranslator:
    """Download Wikipedia dumps and create backtranslation pairs via NLLB."""

    def __init__(self, output_dir: str | None = None):
        from utils.common_utils import RuntimeDirectoryManager
        self.output_dir = Path(output_dir) if output_dir else RuntimeDirectoryManager().raw_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def download_monolingual(
        self,
        langs: Optional[List[str]] = None,
        max_per_lang: int = 500_000,
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, int]:
        if max_concurrent is None:
            max_concurrent = int(os.environ.get("WIKI_DOWNLOAD_WORKERS", "20"))
        """Download Wikipedia for each language and write mono_{lang}.txt files.

        Downloads run in parallel (max_concurrent at a time) since the work is
        IO-bound (streaming datasets + text cleaning).
        """
        if langs is None:
            langs = list(WIKIPEDIA_LANG_CODES.keys())

        results: Dict[str, int] = {}

        def _download_one(lang: str) -> tuple[str, List[str]]:
            sentences = download_wikipedia_corpus(lang, max_sentences=max_per_lang)
            return lang, sentences

        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futmap = {pool.submit(_download_one, lang): lang for lang in langs}
            for fut in as_completed(futmap):
                lang = futmap[fut]
                try:
                    _lang, sentences = fut.result()
                except Exception as e:
                    self.logger.error(f"Wikipedia download failed for {lang}: {e}")
                    results[lang] = 0
                    continue

                if not sentences:
                    self.logger.warning(f"No Wikipedia data for {lang}, skipping")
                    results[lang] = 0
                    continue

                out_path = self.output_dir / f"mono_{lang}.txt"
                with open(out_path, 'w', encoding='utf-8') as f:
                    for sent in sentences:
                        f.write(sent + '\n')
                results[lang] = len(sentences)
                self.logger.info(f"Wrote {len(sentences):,} sentences to {out_path}")

        return results


def main():
    """Standalone: download Wikipedia for all 20 languages."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    bt = WikipediaBacktranslator()
    stats = bt.download_monolingual(max_per_lang=200_000)
    for lang, count in stats.items():
        print(f"  {lang}: {count:,}")


if __name__ == "__main__":
    main()
