# tools/acquire_domain_data.py

import logging
from pathlib import Path
from datasets import load_dataset, get_dataset_split_names
from tqdm import tqdm
import argparse

# Use your existing, excellent logging and directory management
from utils.common_utils import DirectoryManager

logger = logging.getLogger(__name__)

class DomainDataAcquirer:
    """
    Downloads and preprocesses domain-specific parallel corpora.
    """
    def __init__(self, output_base_dir: str = "data/raw"):
        self.output_base_dir = Path(output_base_dir)
        DirectoryManager.create_directory(self.output_base_dir)
        logger.info(f"Domain data will be saved in: {self.output_base_dir}")

    def process_and_save(self, dataset, output_path: Path, lang1: str, lang2: str):
        """
        Processes a Hugging Face dataset and saves it in the pipeline's format.
        """
        logger.info(f"Processing and saving to {output_path}...")
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc=f"Writing {output_path.name}"):
                try:
                    source_text = example['translation'][lang1].strip()
                    target_text = example['translation'][lang2].strip()
                    
                    if source_text and target_text:
                        f.write(f"{source_text}\t{target_text}\t{lang1}\t{lang2}\n")
                        count += 1
                except (KeyError, TypeError):
                    logger.warning(f"Skipping malformed record: {example}")
                    continue
        logger.info(f"✅ Saved {count:,} sentence pairs to {output_path}")

    def acquire_medical_data(self, lang_pairs: list[tuple[str, str]]):
        """
        Acquires medical data from the OPUS UFAL Medical Corpus.
        Example: acquire_medical_data([('en', 'es')])
        """
        logger.info("\n--- Acquiring Medical Data (UFAL Medical Corpus) ---")
        domain_dir = DirectoryManager.create_directory(self.output_base_dir / "medical")
        
        for lang1, lang2 in lang_pairs:
            try:
                # The UFAL corpus is part of the OPUS collection
                dataset = load_dataset("opus_medical", f"{lang1}-{lang2}", split="train", trust_remote_code=True)
                output_file = domain_dir / f"{lang1}-{lang2}_medical.txt"
                self.process_and_save(dataset, output_file, lang1, lang2)
            except Exception as e:
                logger.error(f"Could not download medical data for {lang1}-{lang2}. Error: {e}")

    def acquire_legal_data(self, lang_pairs: list[tuple[str, str]]):
        """
        Acquires legal data from the JRC-Acquis corpus.
        Example: acquire_legal_data([('en', 'de'), ('en', 'fr')])
        """
        logger.info("\n--- Acquiring Legal Data (JRC-Acquis) ---")
        domain_dir = DirectoryManager.create_directory(self.output_base_dir / "legal")

        for lang1, lang2 in lang_pairs:
            try:
                # JRC-Acquis has a specific naming convention
                dataset = load_dataset("jrc_acquis", f"{lang1}-{lang2}", split="train", trust_remote_code=True)
                output_file = domain_dir / f"{lang1}-{lang2}_legal.txt"
                self.process_and_save(dataset, output_file, lang1, lang2)
            except Exception as e:
                logger.error(f"Could not download legal data for {lang1}-{lang2}. Error: {e}")

    def acquire_tech_data(self, lang_pairs: list[tuple[str, str]]):
        """
        Acquires tech data from OPUS (GNOME, KDE, Ubuntu localization files).
        Example: acquire_tech_data([('en', 'fr')])
        """
        logger.info("\n--- Acquiring Tech Data (OPUS GNOME/KDE) ---")
        domain_dir = DirectoryManager.create_directory(self.output_base_dir / "tech")
        
        for lang1, lang2 in lang_pairs:
            try:
                # We can combine multiple tech-related corpora
                gnome_dataset = load_dataset("opus_gnome", f"{lang1}-{lang2}", split="train", trust_remote_code=True)
                kde_dataset = load_dataset("opus_kde4", f"{lang1}-{lang2}", split="train", trust_remote_code=True)
                
                output_file = domain_dir / f"{lang1}-{lang2}_tech.txt"
                
                # Process GNOME dataset
                logger.info(f"Processing GNOME dataset for {lang1}-{lang2}...")
                self.process_and_save(gnome_dataset, output_file, lang1, lang2)
                
                # Append KDE data to the same file
                logger.info(f"Processing KDE4 dataset for {lang1}-{lang2}...")
                with open(output_file, 'a', encoding='utf-8') as f:
                    count = 0
                    for example in tqdm(kde_dataset, desc=f"Appending KDE data"):
                        source_text = example['translation'][lang1].strip()
                        target_text = example['translation'][lang2].strip()
                        if source_text and target_text:
                            f.write(f"{source_text}\t{target_text}\t{lang1}\t{lang2}\n")
                            count += 1
                    logger.info(f"✅ Appended {count:,} sentence pairs from KDE4.")

            except Exception as e:
                logger.error(f"Could not download tech data for {lang1}-{lang2}. Error: {e}")

def main():
    """
    Main function to run the data acquisition process from the command line.
    """
    parser = argparse.ArgumentParser(description="Domain-Specific Data Acquirer for Universal Translation System")
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=['medical', 'legal', 'tech', 'all'],
        help="The domain to acquire data for."
    )
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Comma-separated language pairs, e.g., 'en-es,en-fr'."
    )
    args = parser.parse_args()

    acquirer = DomainDataAcquirer()
    lang_pairs = [tuple(pair.split('-')) for pair in args.pairs.split(',')]

    if args.domain == 'medical' or args.domain == 'all':
        acquirer.acquire_medical_data(lang_pairs)
    
    if args.domain == 'legal' or args.domain == 'all':
        acquirer.acquire_legal_data(lang_pairs)

    if args.domain == 'tech' or args.domain == 'all':
        acquirer.acquire_tech_data(lang_pairs)
        
    logger.info("\n🎉 Domain data acquisition complete!")
    logger.info("Run the main data pipeline to process and integrate the new data.")

if __name__ == "__main__":
    # Example Usage from command line:
    # python -m tools.acquire_domain_data --domain medical --pairs "en-es"
    # python -m tools.acquire_domain_data --domain legal --pairs "en-de,en-fr"
    # python -m tools.acquire_domain_data --domain all --pairs "en-es,en-de,en-fr"
    main()
