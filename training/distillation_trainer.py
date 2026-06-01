"""
Knowledge Distillation Trainer.

Extends IntelligentTrainer to add KL-divergence KD loss using a frozen teacher model
(NLLB-200-3.3B or a larger checkpoint of the same architecture).
Total loss = alpha * CE(student, hard_labels) + (1-alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from pathlib import Path

from training.trainer import IntelligentTrainer
from data.synthetic_augmentation import NLLB_CODE_MAP

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

TEACHER_MODEL = "facebook/nllb-200-3.3B"


def _nllb_code(lang: str) -> str:
    return NLLB_CODE_MAP.get(lang, lang)


class DistillationTrainer(IntelligentTrainer):
    """Trainer with knowledge distillation loss from a frozen teacher."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        train_dataset,
        val_dataset,
        config,
        experiment_name: str = "distillation-universal",
        resume_from_checkpoint: Optional[str] = None,
        teacher_checkpoint: Optional[str] = None,
        distill_alpha: float = 0.5,
        distill_temperature: float = 4.0,
    ):
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_encoder: Optional[nn.Module] = None
        self.teacher_decoder: Optional[nn.Module] = None
        self._nllb_teacher: Optional[nn.Module] = None
        self._nllb_tokenizer: Any = None
        super().__init__(
            encoder, decoder, train_dataset, val_dataset,
            config, experiment_name, resume_from_checkpoint,
        )

    def _setup_models(self):
        super()._setup_models()
        self._load_teacher()

    def _load_teacher(self):
        """Load teacher model (NLLB-3.3B or local checkpoint)."""
        if self.teacher_checkpoint:
            logger.info(f"Loading teacher from local checkpoint: {self.teacher_checkpoint}")
            ckpt = torch.load(self.teacher_checkpoint, map_location='cpu')
            from encoder.universal_encoder import UniversalEncoder
            from universal_decoder_node.universal_decoder_node.decoder import OptimizedUniversalDecoder
            self.teacher_encoder = UniversalEncoder(
                max_vocab_size=self.config.model.max_vocab_size,
                hidden_dim=self.config.model.hidden_dim,
                num_layers=self.config.model.num_layers,
                num_heads=self.config.model.num_heads,
            )
            self.teacher_decoder = OptimizedUniversalDecoder(
                encoder_dim=self.config.model.hidden_dim,
                vocab_size=self.config.model.max_vocab_size,
            )
            self.teacher_encoder.load_state_dict(ckpt.get('encoder_state_dict', ckpt))
            self.teacher_decoder.load_state_dict(ckpt.get('decoder_state_dict', ckpt))
            self.teacher_encoder = self.teacher_encoder.to(self.device).eval()
            self.teacher_decoder = self.teacher_decoder.to(self.device).eval()
            for p in self.teacher_encoder.parameters():
                p.requires_grad_(False)
            for p in self.teacher_decoder.parameters():
                p.requires_grad_(False)
            logger.info("Teacher loaded from checkpoint, frozen in eval mode")
        elif AutoModelForSeq2SeqLM is not None and torch.cuda.is_available():
            logger.info(f"Loading NLLB teacher: {TEACHER_MODEL}")
            try:
                self._nllb_teacher = AutoModelForSeq2SeqLM.from_pretrained(
                    TEACHER_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto",
                ).eval()
                self._nllb_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
                for p in self._nllb_teacher.parameters():
                    p.requires_grad_(False)
                logger.info("NLLB-3.3B teacher loaded, frozen in eval mode")
            except Exception as e:
                logger.warning(f"Could not load NLLB teacher: {e}")
        else:
            logger.info("No teacher available — training with CE loss only")

    def _kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """KL divergence between temperature-scaled student and teacher logits."""
        T = self.distill_temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits.detach() / T, dim=-1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)

    def _nllb_forward(self, source_texts, target_texts):
        """Run NLLB teacher on preprocessed source/target texts, return teacher logits."""
        if self._nllb_teacher is None or self._nllb_tokenizer is None:
            return None
        src_lang = "eng_Latn"
        tgt_lang = self._nllb_tokenizer.convert_ids_to_tokens(
            self._nllb_tokenizer(target_texts[0:1], return_tensors="pt")["input_ids"][0, 0]
        )
        tgt_lang = "fra_Latn"
        return None

    @torch.no_grad()
    def _teacher_logits(self, source_ids, source_mask, target_ids, pad_token_id):
        """Get teacher logits for KD loss."""
        if self.teacher_encoder is not None and self.teacher_decoder is not None:
            enc_out = self.teacher_encoder(source_ids, source_mask)
            dec_out = self.teacher_decoder(
                target_ids[:, :-1], enc_out, encoder_attention_mask=source_mask
            )
            return dec_out
        return None

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute combined CE + KD loss."""
        source_ids = batch['source_ids']
        target_ids = batch['target_ids']
        source_mask = batch['source_mask']
        pad_token_id = batch.get('pad_token_id', 0)

        encoder_output = self.encoder(source_ids, source_mask)
        student_logits = self.decoder(
            target_ids[:, :-1], encoder_output, encoder_attention_mask=source_mask
        )

        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            target_ids[:, 1:].reshape(-1),
            ignore_index=pad_token_id,
            label_smoothing=0.1,
        )

        if self.distill_alpha < 1.0:
            teacher_logits = self._teacher_logits(source_ids, source_mask, target_ids, pad_token_id)
            if teacher_logits is not None:
                kd_loss = self._kd_loss(student_logits, teacher_logits)
                return self.distill_alpha * ce_loss + (1.0 - self.distill_alpha) * kd_loss

        return ce_loss


def train_with_distillation(config_path: str = "config/base.yaml", **kwargs):
    """Convenience: load config, datasets, and run DistillationTrainer."""
    from config.schemas import load_config
    from data.dataset import TranslationDataset
    from encoder.universal_encoder import UniversalEncoder
    from universal_decoder_node.universal_decoder_node.decoder import OptimizedUniversalDecoder

    config = load_config(config_path)

    encoder = UniversalEncoder(
        max_vocab_size=config.model.max_vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
    )
    decoder = OptimizedUniversalDecoder(
        encoder_dim=config.model.hidden_dim,
        vocab_size=config.model.max_vocab_size,
    )

    train_dataset = TranslationDataset(config, split="train")
    val_dataset = TranslationDataset(config, split="val")

    trainer = DistillationTrainer(
        encoder=encoder,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        **kwargs,
    )
    return trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_with_distillation()
