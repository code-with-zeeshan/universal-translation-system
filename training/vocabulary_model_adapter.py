import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cloud_decoder.optimized_decoder import OptimizedUniversalDecoder
from encoder.universal_encoder import UniversalEncoder

logger = logging.getLogger("training.vocab_adapter")


class EmbeddingResizeAdapter:
    """
    Resizes encoder/decoder embeddings for vocabulary evolution and
    runs a short finetuning pass to train the new embedding rows.

    Usage:
        adapter = EmbeddingResizeAdapter(encoder, decoder)
        adapter.resize(
            new_encoder_vocab_size, new_decoder_vocab_size,
            new_token_ids, old_pack, evolved_pack
        )
        adapter.finetune_new_embeddings(dataloader, device='cuda', epochs=3)
        adapter.save_checkpoint(output_path)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.old_encoder_vocab: Optional[int] = None
        self.old_decoder_vocab: Optional[int] = None

    def resize(
        self,
        new_vocab_size: int,
        new_decoder_vocab_size: Optional[int] = None,
    ):
        if new_decoder_vocab_size is None:
            new_decoder_vocab_size = new_vocab_size

        self.old_encoder_vocab = getattr(self.encoder, "vocab_size", None)
        if self.old_encoder_vocab is None:
            self.old_encoder_vocab = self.encoder.embedding_layer.weight.size(0)

        self.old_decoder_vocab = self.decoder.vocab_size

        if new_vocab_size <= self.old_encoder_vocab and new_decoder_vocab_size <= self.old_decoder_vocab:
            logger.info("No resize needed — new vocab size <= existing size")
            return

        logger.info(
            f"Resizing encoder: {self.old_encoder_vocab} -> {new_vocab_size}, "
            f"decoder: {self.old_decoder_vocab} -> {new_decoder_vocab_size}"
        )

        self._resize_encoder_embeddings(new_vocab_size)
        self._resize_decoder_embeddings(new_decoder_vocab_size)

    def _resize_encoder_embeddings(self, new_size: int):
        old_layer = self.encoder.embedding_layer
        old_weight = old_layer.weight.data.clone()
        hidden_dim = old_layer.weight.size(1)

        new_layer = nn.Embedding(new_size, hidden_dim)
        new_layer = new_layer.to(old_weight.device)
        new_layer.weight.data[:self.old_encoder_vocab] = old_weight

        self.encoder.embedding_layer = new_layer
        self.encoder.vocab_size = new_size
        logger.info(f"Encoder embedding resized to [{new_size}, {hidden_dim}]")

    def _resize_decoder_embeddings(self, new_size: int):
        old_embed = self.decoder.embedding
        old_weight = old_embed.weight.data.clone()
        decoder_dim = old_embed.weight.size(1)

        new_embed = nn.Embedding(new_size, decoder_dim)
        new_embed = new_embed.to(old_weight.device)
        new_embed.weight.data[:self.old_decoder_vocab] = old_weight

        old_proj = self.decoder.output_projection
        new_proj = nn.Linear(decoder_dim, new_size, bias=False)
        new_proj = new_proj.to(old_proj.weight.device)
        new_proj.weight = new_embed.weight

        self.decoder.embedding = new_embed
        self.decoder.output_projection = new_proj
        self.decoder.vocab_size = new_size
        logger.info(f"Decoder embedding resized to [{new_size}, {decoder_dim}]")

    def initialize_new_embeddings(
        self,
        new_tokens: List[str],
        encoder_vocab: Dict[str, int],
        decoder_vocab: Dict[str, int],
    ):
        n = max(
            self.encoder.embedding_layer.weight.size(0),
            self.decoder.embedding.weight.size(0),
        )
        if n <= self.old_encoder_vocab and n <= self.old_decoder_vocab:
            return

        new_ids_encoder = []
        new_ids_decoder = []

        encoder_weight = self.encoder.embedding_layer.weight.data
        decoder_weight = self.decoder.embedding.weight.data

        for token in new_tokens:
            enc_id = encoder_vocab.get(token)
            if enc_id is not None and enc_id >= self.old_encoder_vocab:
                new_ids_encoder.append(enc_id)
            dec_id = decoder_vocab.get(token)
            if dec_id is not None and dec_id >= self.old_decoder_vocab:
                new_ids_decoder.append(dec_id)

        self._init_embedding_rows(
            encoder_weight, new_ids_encoder, self.old_encoder_vocab
        )
        self._init_embedding_rows(
            decoder_weight, new_ids_decoder, self.old_decoder_vocab
        )
        logger.info(
            f"Initialized {len(new_ids_encoder)} new encoder and "
            f"{len(new_ids_decoder)} new decoder embedding rows"
        )

    def _init_embedding_rows(
        self,
        weight: torch.Tensor,
        new_ids: List[int],
        old_vocab_size: int,
    ):
        if not new_ids:
            return

        old_weights = weight[:old_vocab_size]
        mean = old_weights.mean(dim=0)
        std = old_weights.std(dim=0)

        noise = torch.randn(len(new_ids), weight.size(1), device=weight.device)
        init_vals = mean.unsqueeze(0) + noise * std.unsqueeze(0)

        for i, idx in enumerate(new_ids):
            weight[idx] = init_vals[i]

    def finetune_new_embeddings(
        self,
        dataloader: DataLoader,
        device: str = "cuda",
        epochs: int = 3,
        lr: float = 1e-4,
    ):
        encoder_embed = self.encoder.embedding_layer
        decoder_embed = self.decoder.embedding

        if self.old_encoder_vocab is not None:
            encoder_new_ids = list(range(self.old_encoder_vocab, encoder_embed.weight.size(0)))
        else:
            encoder_new_ids = []

        if self.old_decoder_vocab is not None:
            decoder_new_ids = list(range(self.old_decoder_vocab, decoder_embed.weight.size(0)))
        else:
            decoder_new_ids = []

        if not encoder_new_ids and not decoder_new_ids:
            logger.info("No new embedding rows to finetune")
            return

        self._freeze_all_except_new(encoder_new_ids, decoder_new_ids)
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.train()
        self.decoder.train()

        params = []
        if encoder_new_ids:
            params.append(encoder_embed.weight)
        if decoder_new_ids:
            params.append(decoder_embed.weight)
        optimizer = torch.optim.Adam(params, lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        total_steps = 0
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                encoder_hidden = self.encoder(input_ids)
                logits = self.decoder(input_ids[:, :-1], encoder_hidden)
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                total_steps += 1

            avg_loss = total_loss / max(num_batches, 1)
            logger.info(
                f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}"
            )

        logger.info(
            f"Finetuning complete: {total_steps} steps, "
            f"trained {len(encoder_new_ids)} encoder + "
            f"{len(decoder_new_ids)} decoder new embeddings"
        )

    def _freeze_all_except_new(
        self,
        encoder_new_ids: List[int],
        decoder_new_ids: List[int],
    ):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        embed = self.encoder.embedding_layer
        embed.weight.requires_grad = False
        for idx in encoder_new_ids:
            embed.weight[idx].requires_grad = True

        embed = self.decoder.embedding
        embed.weight.requires_grad = False
        for idx in decoder_new_ids:
            embed.weight[idx].requires_grad = True

        n_trainable = sum(
            p.numel() for p in list(self.encoder.parameters()) + list(self.decoder.parameters())
            if p.requires_grad
        )
        logger.info(
            f"Frozen all weights except {len(encoder_new_ids)} encoder + "
            f"{len(decoder_new_ids)} decoder new embedding rows "
            f"({n_trainable} trainable parameters)"
        )

    def save_checkpoint(self, output_path: str, metadata: Optional[Dict] = None):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "encoder_vocab_size": getattr(self.encoder, "vocab_size", None),
            "decoder_vocab_size": self.decoder.vocab_size,
        }
        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, str(path))
        logger.info(f"Checkpoint saved to {path}")

    @staticmethod
    def load_checkpoint(
        encoder: nn.Module,
        decoder: nn.Module,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])

        if "encoder_vocab_size" in checkpoint:
            encoder.vocab_size = checkpoint["encoder_vocab_size"]
        if "decoder_vocab_size" in checkpoint:
            decoder.vocab_size = checkpoint["decoder_vocab_size"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get("metadata")
