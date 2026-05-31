"""
PEFT (LoRA) integration for efficient adapter training.
Wraps encoder/decoder with LoRA adapters, freezing the backbone.
"""
import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def wrap_encoder_with_lora(
    encoder: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    use_rslora: bool = True,
) -> nn.Module:
    """Wrap encoder with LoRA adapters using PEFT.
    
    Args:
        encoder: The encoder model to wrap
        r: LoRA rank
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout for LoRA layers
        target_modules: Module names to apply LoRA to (auto-detected if None)
        use_rslora: Use Rank-Stabilized LoRA (better quality at low rank)
    
    Returns:
        PEFT-wrapped encoder with frozen backbone + trainable LoRA adapters
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.warning("peft not installed, falling back to manual adapter training")
        return encoder

    if target_modules is None:
        target_modules = _detect_linear_modules(encoder)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        use_rslora=use_rslora,
    )

    try:
        encoder = get_peft_model(encoder, lora_config)
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in encoder.parameters())
        logger.info(f"Encoder LoRA: {trainable:,}/{total:,} params trainable ({100*trainable/total:.1f}%)")
        return encoder
    except Exception as e:
        logger.warning(f"PEFT wrapping failed ({e}), falling back to manual freeze")
        return _freeze_backbone_manual(encoder)


def wrap_decoder_with_lora(
    decoder: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    use_rslora: bool = True,
) -> nn.Module:
    """Wrap decoder with LoRA adapters using PEFT.
    
    Args:
        decoder: The decoder model to wrap
        r: LoRA rank
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout for LoRA layers
        target_modules: Module names to apply LoRA to (auto-detected if None)
        use_rslora: Use Rank-Stabilized LoRA
    
    Returns:
        PEFT-wrapped decoder with frozen backbone + trainable LoRA adapters
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.warning("peft not installed, falling back to manual adapter training")
        return decoder

    if target_modules is None:
        target_modules = _detect_linear_modules(decoder)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        use_rslora=use_rslora,
    )

    try:
        decoder = get_peft_model(decoder, lora_config)
        trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in decoder.parameters())
        logger.info(f"Decoder LoRA: {trainable:,}/{total:,} params trainable ({100*trainable/total:.1f}%)")
        return decoder
    except Exception as e:
        logger.warning(f"PEFT wrapping failed ({e}), falling back to manual freeze")
        return _freeze_backbone_manual(decoder)


def _detect_linear_modules(model: nn.Module) -> list:
    """Auto-detect linear module names for LoRA application."""
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Extract the leaf module name
            parts = name.split(".")
            if parts:
                leaf = parts[-1]
                if leaf not in linear_modules:
                    linear_modules.append(leaf)
    logger.info(f"Auto-detected LoRA target modules: {linear_modules}")
    return linear_modules


def _freeze_backbone_manual(model: nn.Module) -> nn.Module:
    """Manual fallback: freeze all backbone params (no PEFT)."""
    for param in model.parameters():
        param.requires_grad = False
    logger.info("All backbone parameters frozen (manual fallback)")
    return model


def get_lora_trainable_params(model: nn.Module) -> list:
    """Get only LoRA/trainable parameters for optimizer."""
    return [p for p in model.parameters() if p.requires_grad]


def print_trainable_parameter_stats(encoder: nn.Module, decoder: nn.Module):
    """Log trainable parameter statistics."""
    encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    encoder_total = sum(p.numel() for p in encoder.parameters())
    decoder_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    decoder_total = sum(p.numel() for p in decoder.parameters())

    logger.info("=" * 60)
    logger.info("TRAINABLE PARAMETER SUMMARY (LoRA)")
    logger.info(f"  Encoder: {encoder_trainable:,}/{encoder_total:,} ({100*encoder_trainable/encoder_total:.1f}%)")
    logger.info(f"  Decoder: {decoder_trainable:,}/{decoder_total:,} ({100*decoder_trainable/decoder_total:.1f}%)")
    logger.info(f"  Total:   {encoder_trainable+decoder_trainable:,}/{encoder_total+decoder_total:,} ({100*(encoder_trainable+decoder_trainable)/(encoder_total+decoder_total):.1f}%)")
    logger.info("=" * 60)
