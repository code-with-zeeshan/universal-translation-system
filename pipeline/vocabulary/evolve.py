import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import msgpack

from utils.common_utils import RuntimeDirectoryManager

from pipeline.vocabulary.creator import UnifiedVocabularyCreator as VocabularyPackCreator

logger = logging.getLogger("vocabulary.evolve")


class VocabularyEvolver:
    """
    Evolves vocabulary packs by promoting frequently-observed unknown tokens.

    Flow:
        1. Collect unknown token analytics (Redis sorted set or JSON file)
        2. Filter tokens above promotion_threshold
        3. For each affected language group, create a new evolved pack version
        4. Optionally trigger model embedding retraining for the new tokens
    """

    def __init__(
        self,
        vocab_dir: str = "",
        promotion_threshold: int = 1000,
    ):
        self.runtime_dirs = RuntimeDirectoryManager()
        resolved_dir = vocab_dir if vocab_dir else str(self.runtime_dirs.vocab_dir)
        self.vocab_dir = Path(resolved_dir)
        self.pack_creator = VocabularyPackCreator(output_dir=resolved_dir)
        self.promotion_threshold = promotion_threshold

    def evolve_all_packs(self) -> Dict[str, int]:
        logger.info("Starting vocabulary evolution process...")

        unknowns = self._load_unknown_tokens()

        tokens_to_promote = {
            token: count
            for token, count in unknowns.items()
            if count >= self.promotion_threshold
        }

        if not tokens_to_promote:
            logger.info("No tokens meet the promotion threshold. Vocabulary is up to date.")
            return {}

        logger.info(f"Found {len(tokens_to_promote)} tokens to promote")

        pack_tokens = self._assign_tokens_to_packs(tokens_to_promote)

        results = {}
        for pack_name, tokens in pack_tokens.items():
            if tokens:
                result = self.evolve_pack(pack_name, tokens)
                results[pack_name] = result

        return results

    def _load_unknown_tokens(self) -> Dict[str, int]:
        try:
            from utils.redis_manager import RedisManager
            rm = RedisManager.get_instance()
            client = rm.get_client()
            if client:
                zitems = client.zrevrange("unknown_token_counts", 0, 1000, withscores=True)
                return {
                    (k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)): int(v)
                    for k, v in zitems
                }
        except Exception as e:
            logger.warning(f"Redis analytics unavailable: {e}")

        analytics_path = os.environ.get("EVOLVE_ANALYTICS_JSON")
        if analytics_path and Path(analytics_path).exists():
            try:
                with open(analytics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("unknown_token_counts", {})
                return {str(k): int(v) for k, v in raw.items()}
            except Exception as e:
                logger.warning(f"Failed loading analytics JSON {analytics_path}: {e}")

        logger.warning(
            "No analytics source available. "
            "Set EVOLVE_ANALYTICS_JSON or configure Redis."
        )
        return {}

    def _assign_tokens_to_packs(
        self, tokens: Dict[str, int]
    ) -> Dict[str, List[str]]:
        packs = self._discover_packs()
        if not packs:
            logger.warning("No existing vocabulary packs found to evolve")
            return {}

        # Load pack languages to route tokens correctly
        pack_languages = {}
        for pack_name, info in packs.items():
            pack_languages[pack_name] = self._load_pack_languages(info["file"])

        # Build reverse mapping: language → pack_name
        lang_to_pack = {}
        for pack_name, langs in pack_languages.items():
            for lang in langs:
                lang_to_pack[lang] = pack_name

        # If we can't determine language from analytics, assign to latin
        default_pack = "latin" if "latin" in packs else next(iter(packs))

        token_list = list(tokens.keys())
        result = {pn: [] for pn in packs}

        for token in token_list:
            # Heuristic: try to guess language from character script
            assigned = False
            for lang, pack_name in lang_to_pack.items():
                if self._token_belongs_to_language(token, lang):
                    result[pack_name].append(token)
                    assigned = True
                    break
            if not assigned:
                result[default_pack].append(token)

        for pack_name, assigned_tokens in result.items():
            if assigned_tokens:
                logger.info(
                    f"Assigned {len(assigned_tokens)} tokens to pack '{pack_name}' "
                    f"(languages: {pack_languages.get(pack_name, [])})"
                )

        return result

    @staticmethod
    def _token_belongs_to_language(token: str, lang: str) -> bool:
        """Heuristic script-based check: does a token likely belong to a language group?"""
        if not token:
            return False
        first_char = token[0]
        cp = ord(first_char)

        # Latin-based languages
        latin_scripts = {'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'pl', 'id', 'vi', 'tr'}
        # CJK
        cjk_scripts = {'zh', 'ja', 'ko'}
        # Arabic
        arabic_scripts = {'ar'}
        # Devanagari
        devanagari_scripts = {'hi'}
        # Cyrillic
        cyrillic_scripts = {'ru', 'uk'}
        # Thai
        thai_scripts = {'th'}

        if lang in latin_scripts:
            return 0x0020 <= cp < 0x007F or 0x00C0 <= cp < 0x024F or 0x1E00 <= cp < 0x1EFF
        if lang in cjk_scripts:
            return 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF
        if lang in arabic_scripts:
            return 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F
        if lang in devanagari_scripts:
            return 0x0900 <= cp <= 0x097F
        if lang in cyrillic_scripts:
            return 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F
        if lang in thai_scripts:
            return 0x0E00 <= cp <= 0x0E7F

        return False

    def _discover_packs(self) -> Dict[str, dict]:
        packs = {}
        for f in self.vocab_dir.glob("*_v*.msgpack"):
            stem = f.stem
            if "_v" in stem:
                pack_name = stem.rsplit("_v", 1)[0]
                if pack_name not in packs or stem > packs[pack_name]["stem"]:
                    packs[pack_name] = {"file": f, "stem": stem}
        return packs

    def _load_pack_languages(self, pack_path: Path) -> List[str]:
        try:
            with open(pack_path, "rb") as f:
                pack = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
            return pack.get("languages", [])
        except Exception:
            return []

    def evolve_pack(self, pack_name: str, new_tokens: List[str]) -> int:
        packs = self._discover_packs()
        if pack_name not in packs:
            logger.error(f"No existing pack found for '{pack_name}'. Cannot evolve.")
            return 0

        base_pack_path = packs[pack_name]["file"]
        languages = self._load_pack_languages(base_pack_path)

        logger.info(
            f"Evolving pack '{pack_name}' with {len(new_tokens)} new tokens "
            f"for languages: {languages}"
        )

        self.pack_creator.create_pack(
            pack_name=pack_name,
            languages=languages,
            base_pack_path=str(base_pack_path),
            tokens_to_add=new_tokens,
        )

        logger.info(f"Successfully created new version for pack '{pack_name}'.")
        return len(new_tokens)


def _build_evolution_dataset(
    new_tokens: List[str],
    token_to_id: Dict[str, int],
    seq_length: int = 16,
    samples_per_token: int = 5,
) -> List[Dict[str, torch.Tensor]]:
    templates = [
        "{} .",
        "the term {} means",
        "{} is a word",
        "translate {}",
        "{} and more",
    ]
    data = []
    for token in new_tokens:
        token_id = token_to_id.get(token)
        if token_id is None:
            continue
        for i in range(samples_per_token):
            tpl = templates[i % len(templates)]
            text = tpl.format(token)
            ids = [token_to_id.get(w, 1) for w in text.split()]
            ids = [token_to_id.get("<s>", 2)] + ids + [token_to_id.get("</s>", 3)]
            ids = ids[:seq_length]
            pad_len = seq_length - len(ids)
            ids = ids + [token_to_id.get("<pad>", 0)] * pad_len
            input_ids = torch.tensor(ids, dtype=torch.long)
            labels = input_ids.clone()
            data.append({"input_ids": input_ids, "labels": labels})
    return data


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evolve vocabulary packs")
    parser.add_argument(
        "--vocab-dir", default=str(RuntimeDirectoryManager().vocab_dir), help="Vocabulary directory"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1000,
        help="Minimum unknown token count to promote",
    )
    parser.add_argument(
        "--pack", default=None, help="Specific pack to evolve (default: all)"
    )
    parser.add_argument(
        "--retrain-model",
        default=None,
        help="Path to model checkpoint to retrain after evolution",
    )
    parser.add_argument(
        "--retrain-epochs",
        type=int,
        default=3,
        help="Number of finetuning epochs for new embeddings",
    )
    parser.add_argument(
        "--retrain-lr",
        type=float,
        default=1e-4,
        help="Learning rate for embedding finetuning",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="Output path for retrained model checkpoint",
    )
    args = parser.parse_args()

    evolver = VocabularyEvolver(
        vocab_dir=args.vocab_dir,
        promotion_threshold=args.threshold,
    )

    new_tokens = list(evolver._load_unknown_tokens().keys())
    if args.pack:
        packs = evolver._discover_packs()
        if args.pack not in packs:
            logger.error(f"Pack '{args.pack}' not found in {args.vocab_dir}")
            raise SystemExit(1)
        token_map = {args.pack: new_tokens}
    else:
        packs = evolver._discover_packs()
        token_map = {p: new_tokens for p in packs}

    total_evolved = 0
    for pack_name, tokens in token_map.items():
        if not tokens:
            continue
        result = evolver.evolve_pack(pack_name, tokens)
        total_evolved += result

    logger.info(
        f"Evolution complete. {total_evolved} tokens promoted "
        f"across {len(token_map)} packs"
    )

    if args.retrain_model and total_evolved > 0:
        logger.info("Starting model retraining for new embeddings...")
        from tools.vocab_adapter import EmbeddingResizeAdapter

        encoder = UniversalEncoder(
            max_vocab_size=50000,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
        )
        decoder = OptimizedUniversalDecoder(
            encoder_dim=384,
            decoder_dim=768,
            vocab_size=50000,
            num_layers=8,
            num_heads=12,
            dropout=0.1,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(args.retrain_model, map_location=device)
        if "encoder_state_dict" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            decoder.load_state_dict(checkpoint["decoder_state_dict"])
        else:
            encoder.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            decoder.load_state_dict(checkpoint.get("decoder_state_dict", checkpoint))

        new_vocab_size = encoder.embedding_layer.weight.size(0) + total_evolved
        new_decoder_size = decoder.embedding.weight.size(0) + total_evolved

        adapter = EmbeddingResizeAdapter(encoder, decoder)
        adapter.resize(new_vocab_size, new_decoder_size)

        for pack_name, tokens in token_map.items():
            pack_path = None
            for f in Path(args.vocab_dir).glob(f"{pack_name}_v*.msgpack"):
                pack_path = f
            if pack_path is None:
                continue
            with open(pack_path, "rb") as f:
                pack_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
            encoder_vocab = {
                **pack_data.get("tokens", {}),
                **pack_data.get("subwords", {}),
                **pack_data.get("special_tokens", {}),
            }
            decoder_vocab = encoder_vocab
            adapter.initialize_new_embeddings(
                tokens, encoder_vocab, decoder_vocab
            )

        dataset_data = []
        for pack_name, tokens in token_map.items():
            pack_path = None
            for f in Path(args.vocab_dir).glob(f"{pack_name}_v*.msgpack"):
                pack_path = f
            if pack_path is None:
                continue
            with open(pack_path, "rb") as f:
                pack_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
            full_vocab = {
                **pack_data.get("tokens", {}),
                **pack_data.get("subwords", {}),
                **pack_data.get("special_tokens", {}),
            }
            dataset_data.extend(
                _build_evolution_dataset(tokens, full_vocab)
            )

        if dataset_data:
            class EvolutionDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    return self.data[idx]

            def collate_fn(batch):
                return {
                    "input_ids": torch.stack([b["input_ids"] for b in batch]),
                    "labels": torch.stack([b["labels"] for b in batch]),
                }

            loader = DataLoader(
                EvolutionDataset(dataset_data),
                batch_size=16,
                shuffle=True,
                collate_fn=collate_fn,
            )

            adapter.finetune_new_embeddings(
                loader, device=device, epochs=args.retrain_epochs, lr=args.retrain_lr
            )

        output_path = args.save_model or args.retrain_model
        adapter.save_checkpoint(output_path, metadata={
            "evolved_tokens": total_evolved,
            "packs_evolved": list(token_map.keys()),
        })
        logger.info(f"Retrained model saved to {output_path}")
