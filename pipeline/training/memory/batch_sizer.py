import torch
import logging
import time

logger = logging.getLogger(__name__)


class DynamicBatchSizer:
    """Dynamic batch sizing based on memory usage and startup probing"""

    def __init__(self, initial_batch_size: int = 32, max_batch_size: int = 128):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = 4
        self.memory_threshold = 0.9
        self.adjustment_factor = 1.2

    def probe(
        self,
        encoder,
        decoder,
        device,
        seq_len: int = 150,
        vocab_size: int = 32000,
    ) -> int:
        """Find max batch size that fits in GPU memory via dummy forward/backward.

        Probe range scales dynamically with GPU memory so high-end cards
        (A100 80GB, H100) find their true limit rather than a hardcoded cap.
        Compile is temporarily disabled during probe to avoid 11× re-trace
        overhead — each probe batch size is a different tensor shape that
        would trigger a fresh compilation.
        """
        import torch._dynamo
        torch._dynamo.reset()

        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        probe_limit = max(128, min(4096, int(total_gb * 10)))
        step = max(4, probe_limit // 40)
        step = (step // 4) * 4
        probe_sizes = list(range(4, probe_limit + 1, step))
        if probe_sizes[-1] < probe_limit:
            probe_sizes.append(probe_limit)
        max_viable = self.min_batch_size

        encoder.train()
        decoder.train()
        dtype = next(iter(device.type == 'cuda' and [torch.bfloat16] or [torch.float32]))

        with torch._dynamo.config.patch(disable=True):
            for bs in probe_sizes:
                if bs <= max_viable:
                    continue
                try:
                    dummy_src = torch.randint(0, min(vocab_size, 1000), (bs, seq_len), device=device)
                    dummy_tgt = torch.randint(0, min(vocab_size, 1000), (bs, seq_len), device=device)
                    dummy_mask = torch.ones(bs, seq_len, dtype=torch.bool, device=device)

                    with torch.amp.autocast(device_type=device.type, dtype=dtype):
                        enc_out = encoder(dummy_src, dummy_mask)
                        dec_out = decoder(
                            decoder_input_ids=dummy_tgt[:, :-1],
                            encoder_hidden_states=enc_out,
                            encoder_attention_mask=dummy_mask,
                        )
                        loss = dec_out.mean()

                    loss.backward()
                    encoder.zero_grad(set_to_none=True)
                    decoder.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(device)

                    max_viable = bs
                    logger.info(f"  ✅ Probe batch_size={bs}: OK")
                except Exception as e:
                    logger.info(f"  ❌ Probe batch_size={bs}: FAILED ({e})")
                    break
                finally:
                    torch.cuda.empty_cache()

        self.current_batch_size = max_viable
        self.max_batch_size = int(max_viable * 1.5)
        logger.info(f"📊 Probed max batch size: {max_viable} (GPU: {total_gb:.0f}GB)")
        return max_viable

    def adjust_batch_size(self) -> int:
        """Adjust batch size based on memory usage"""
        if not torch.cuda.is_available():
            return self.current_batch_size

        memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

        if memory_usage > self.memory_threshold:
            new_size = max(self.min_batch_size, int(self.current_batch_size / self.adjustment_factor))
            if new_size != self.current_batch_size:
                logger.info(f"🔽 Reducing batch size: {self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size
        elif memory_usage < 0.7:
            new_size = min(self.max_batch_size, int(self.current_batch_size * self.adjustment_factor))
            if new_size != self.current_batch_size:
                logger.info(f"🔼 Increasing batch size: {self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size

        return self.current_batch_size
