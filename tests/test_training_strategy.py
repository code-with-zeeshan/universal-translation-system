"""
Tests for training.training_strategy - training strategy dataclass.
"""

import torch
import pytest
from pipeline.training.strategy import TrainingStrategy
from pipeline.training.hardware import HardwareProfile
from pipeline.training.memory.config import MemoryConfig


class TestTrainingStrategy:
    def test_create_minimal(self):
        strategy = TrainingStrategy(
            hardware_profile=HardwareProfile.HIGH_END_SINGLE,
            use_distributed=False,
            distributed_backend="nccl",
            use_fsdp=False,
            use_ddp=False,
            memory_config=MemoryConfig(),
            batch_size=32,
            accumulation_steps=1,
            compile_mode="reduce-overhead",
            mixed_precision_dtype=torch.bfloat16,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            gradient_clipping=1.0,
            learning_rate=5e-5,
            warmup_steps=1000,
        )
        assert strategy.hardware_profile == HardwareProfile.HIGH_END_SINGLE
        assert strategy.use_distributed is False
        assert strategy.use_fsdp is False
        assert strategy.use_ddp is False
        assert strategy.batch_size == 32
        assert strategy.accumulation_steps == 1
        assert strategy.compile_mode == "reduce-overhead"
        assert strategy.mixed_precision_dtype == torch.bfloat16
        assert strategy.num_workers == 4
        assert strategy.pin_memory is True
        assert strategy.prefetch_factor == 2
        assert strategy.gradient_clipping == 1.0
        assert strategy.learning_rate == 5e-5
        assert strategy.warmup_steps == 1000

    def test_ddp_config(self):
        strategy = TrainingStrategy(
            hardware_profile=HardwareProfile.MID_RANGE_MULTI,
            use_distributed=True, distributed_backend="nccl",
            use_fsdp=False, use_ddp=True,
            memory_config=MemoryConfig(),
            batch_size=64, accumulation_steps=2,
            compile_mode="default", mixed_precision_dtype=torch.float16,
            num_workers=8, pin_memory=True, prefetch_factor=4,
            gradient_clipping=0.5, learning_rate=1e-4, warmup_steps=500,
        )
        assert strategy.use_distributed is True
        assert strategy.use_ddp is True
        assert strategy.use_fsdp is False

    def test_fsdp_config(self):
        strategy = TrainingStrategy(
            hardware_profile=HardwareProfile.HIGH_END_MULTI,
            use_distributed=True, distributed_backend="nccl",
            use_fsdp=True, use_ddp=False,
            memory_config=MemoryConfig(),
            batch_size=128, accumulation_steps=4,
            compile_mode="max-autotune", mixed_precision_dtype=torch.bfloat16,
            num_workers=8, pin_memory=True, prefetch_factor=4,
            gradient_clipping=1.0, learning_rate=3e-5, warmup_steps=2000,
        )
        assert strategy.use_distributed is True
        assert strategy.use_fsdp is True
        assert strategy.use_ddp is False

    def test_cpu_only(self):
        strategy = TrainingStrategy(
            hardware_profile=HardwareProfile.CPU_ONLY,
            use_distributed=False, distributed_backend="gloo",
            use_fsdp=False, use_ddp=False,
            memory_config=MemoryConfig(gradient_checkpointing=False, mixed_precision=False, cpu_offload=False, pin_memory=False),
            batch_size=8, accumulation_steps=1,
            compile_mode="default", mixed_precision_dtype=torch.float32,
            num_workers=2, pin_memory=False, prefetch_factor=1,
            gradient_clipping=1.0, learning_rate=1e-4, warmup_steps=100,
        )
        assert strategy.hardware_profile == HardwareProfile.CPU_ONLY
        assert strategy.distributed_backend == "gloo"
        assert strategy.mixed_precision_dtype == torch.float32
        assert strategy.pin_memory is False

    def test_apple_silicon(self):
        strategy = TrainingStrategy(
            hardware_profile=HardwareProfile.APPLE_SILICON,
            use_distributed=False, distributed_backend="mps",
            use_fsdp=False, use_ddp=False,
            memory_config=MemoryConfig(),
            batch_size=16, accumulation_steps=2,
            compile_mode="reduce-overhead", mixed_precision_dtype=torch.float16,
            num_workers=4, pin_memory=False, prefetch_factor=2,
            gradient_clipping=1.0, learning_rate=5e-5, warmup_steps=500,
        )
        assert strategy.hardware_profile == HardwareProfile.APPLE_SILICON
        assert strategy.distributed_backend == "mps"

    def test_low_end_single_gpu(self):
        strategy = TrainingStrategy(
            hardware_profile=HardwareProfile.LOW_END_SINGLE,
            use_distributed=False, distributed_backend="nccl",
            use_fsdp=False, use_ddp=False,
            memory_config=MemoryConfig(),
            batch_size=16, accumulation_steps=4,
            compile_mode="reduce-overhead", mixed_precision_dtype=torch.float16,
            num_workers=2, pin_memory=True, prefetch_factor=2,
            gradient_clipping=1.0, learning_rate=2e-5, warmup_steps=1000,
        )
        assert strategy.use_distributed is False
        assert strategy.use_fsdp is False
        assert strategy.use_ddp is False

    def test_all_hardware_profiles(self):
        for profile in HardwareProfile:
            strategy = TrainingStrategy(
                hardware_profile=profile,
                use_distributed=False, distributed_backend="nccl",
                use_fsdp=False, use_ddp=False,
                memory_config=MemoryConfig(),
                batch_size=32, accumulation_steps=1,
                compile_mode="default", mixed_precision_dtype=torch.float32,
                num_workers=2, pin_memory=True, prefetch_factor=2,
                gradient_clipping=1.0, learning_rate=1e-4, warmup_steps=500,
            )
            assert strategy.hardware_profile == profile

    def test_field_types(self):
        strategy = TrainingStrategy(
            hardware_profile=HardwareProfile.TPU,
            use_distributed=True, distributed_backend="gloo",
            use_fsdp=True, use_ddp=False,
            memory_config=MemoryConfig(),
            batch_size=256, accumulation_steps=1,
            compile_mode="max-autotune", mixed_precision_dtype=torch.bfloat16,
            num_workers=16, pin_memory=True, prefetch_factor=8,
            gradient_clipping=0.0, learning_rate=1e-3, warmup_steps=10000,
        )
        assert isinstance(strategy.hardware_profile, HardwareProfile)
        assert isinstance(strategy.use_distributed, bool)
        assert isinstance(strategy.distributed_backend, str)
        assert isinstance(strategy.use_fsdp, bool)
        assert isinstance(strategy.use_ddp, bool)
        assert isinstance(strategy.batch_size, int)
        assert isinstance(strategy.accumulation_steps, int)
        assert isinstance(strategy.compile_mode, str)
        assert isinstance(strategy.mixed_precision_dtype, torch.dtype)
        assert isinstance(strategy.num_workers, int)
        assert isinstance(strategy.pin_memory, bool)
        assert isinstance(strategy.prefetch_factor, int)
        assert isinstance(strategy.gradient_clipping, float)
        assert isinstance(strategy.learning_rate, float)
        assert isinstance(strategy.warmup_steps, int)

    def test_mixed_precision_dtypes(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            strategy = TrainingStrategy(
                hardware_profile=HardwareProfile.HIGH_END_SINGLE,
                use_distributed=False, distributed_backend="nccl",
                use_fsdp=False, use_ddp=False,
                memory_config=MemoryConfig(),
                batch_size=32, accumulation_steps=1,
                compile_mode="default", mixed_precision_dtype=dtype,
                num_workers=2, pin_memory=True, prefetch_factor=2,
                gradient_clipping=1.0, learning_rate=1e-4, warmup_steps=500,
            )
            assert strategy.mixed_precision_dtype == dtype

    def test_compile_modes(self):
        for mode in ["default", "reduce-overhead", "max-autotune"]:
            strategy = TrainingStrategy(
                hardware_profile=HardwareProfile.HIGH_END_SINGLE,
                use_distributed=False, distributed_backend="nccl",
                use_fsdp=False, use_ddp=False,
                memory_config=MemoryConfig(),
                batch_size=32, accumulation_steps=1,
                compile_mode=mode, mixed_precision_dtype=torch.float16,
                num_workers=2, pin_memory=True, prefetch_factor=2,
                gradient_clipping=1.0, learning_rate=1e-4, warmup_steps=500,
            )
            assert strategy.compile_mode == mode
