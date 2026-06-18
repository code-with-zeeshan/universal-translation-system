"""
Knowledge distillation from NLLB-200-1.3B teacher.

Retranslates existing parallel data with the 1.3B distilled model
to produce higher-quality soft targets for training.
Uses the same distilled model as augment to avoid loading two NLLB variants.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from pipeline.data.augmentation import NLLB_CODE_MAP
from utils.gpu_utils import get_gpu_profile, configure_nllb_model

logger = logging.getLogger(__name__)

TEACHER_MODEL = "facebook/nllb-200-distilled-1.3B"

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    import torch
except Exception:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    pipeline = None
    torch = None


def _lazy_teacher(model_name: str = TEACHER_MODEL):
    """Lazy-load teacher model with GPU-tier optimizations."""
    if AutoModelForSeq2SeqLM is None:
        raise ImportError("transformers required for distillation")
    gpu = get_gpu_profile()
    logger.info(f"Loading teacher model {model_name} on {gpu.name}...")
    use_cuda = torch is not None and torch.cuda.is_available()
    if use_cuda:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype.get(gpu.nllb_dtype, torch.float16)
        use_fa2 = gpu.nllb_flash_attention_2
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation="flash_attention_2" if use_fa2 else "eager",
        )
        model = configure_nllb_model(model, gpu)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def _nllb_code(lang: str) -> str:
    return NLLB_CODE_MAP.get(lang, lang)


class KnowledgeDistillator:
    """Re-translate existing parallel data with NLLB-1.3B teacher."""

    def __init__(self, model_name: str = TEACHER_MODEL):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._translator = None
        self.logger = logging.getLogger(__name__)

    @property
    def translator(self):
        if self._translator is None:
            model, tokenizer = _lazy_teacher(self._model_name)
            gpu = get_gpu_profile()
            use_cuda = torch is not None and torch.cuda.is_available()
            bs = gpu.nllb_batch_size if use_cuda else 1
            self.logger.info(f"Creating distillation pipeline batch_size={bs} (profile: {gpu.name})")
            pipe_kwargs = dict(
                task="translation",
                model=model,
                tokenizer=tokenizer,
                batch_size=bs,
            )
            if not use_cuda:
                pipe_kwargs["device"] = -1
            self._translator = pipeline(**pipe_kwargs)
        return self._translator

    def distill_parallel_file(
        self,
        input_file: str,
        output_file: str,
        source_lang: str,
        target_lang: str,
        max_pairs: int = 100_000,
    ) -> int:
        """Read parallel data, re-translate source side with teacher, write distilled pairs."""
        input_path = Path(input_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            self.logger.error(f"Input not found: {input_path}")
            return 0

        pairs: List[Tuple[str, str]] = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
                if len(pairs) >= max_pairs:
                    break

        if not pairs:
            return 0

        self.logger.info(f"Distilling {len(pairs):,} pairs {source_lang}→{target_lang}...")
        sources = [src for src, _ in pairs]

        count = 0
        batch_size = 32
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for i in tqdm(range(0, len(sources), batch_size), desc=f"Distill {source_lang}→{target_lang}"):
                batch = sources[i:i + batch_size]
                try:
                    results = self.translator(
                        batch,
                        src_lang=_nllb_code(source_lang),
                        tgt_lang=_nllb_code(target_lang),
                        max_length=512,
                    )
                    for src_text, r in zip(batch, results):
                        teacher_tgt = r['translation_text']
                        if teacher_tgt:
                            f_out.write(f"{src_text}\t{teacher_tgt}\n")
                            count += 1
                except Exception as e:
                    self.logger.error(f"Batch distillation failed: {e}")
                    continue

        self.logger.info(f"Distilled {count:,} pairs to {output_path}")
        return count

    def distill_sampled_dir(
        self,
        sampled_dir: str,
        output_dir: str,
        max_pairs_per_pair: int = 50_000,
    ) -> Dict[str, int]:
        """Distill all *_{src}_{tgt}_sampled.txt files in a directory."""
        sampled_path = Path(sampled_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results: Dict[str, int] = {}
        for fpath in sorted(sampled_path.glob("*_sampled.txt")):
            stem = fpath.stem.replace("_sampled", "")
            if "_" not in stem:
                continue
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            src, tgt = parts[0], parts[1]
            out_file = output_path / f"distilled_{src}_{tgt}.txt"
            count = self.distill_parallel_file(
                str(fpath), str(out_file),
                source_lang=src, target_lang=tgt,
                max_pairs=max_pairs_per_pair,
            )
            results[f"{src}_{tgt}"] = count
        return results


def main():
    """Standalone: distill sampled data."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    distiller = KnowledgeDistillator()
    from utils.common_utils import RuntimeDirectoryManager
    rdm = RuntimeDirectoryManager()
    stats = distiller.distill_sampled_dir(str(rdm.sampled_dir), str(rdm.distilled_dir))
    for pair, count in stats.items():
        print(f"  {pair}: {count:,}")


if __name__ == "__main__":
    main()
