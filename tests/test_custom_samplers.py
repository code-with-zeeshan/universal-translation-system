"""
Tests for data.custom_samplers - custom sampling strategies.
"""

import pytest
import torch
from torch.utils.data import Dataset
from pipeline.training.samplers import TemperatureSampler


class MockDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def get_lang_pair_indices(self):
        return {"en-fr": list(range(60)), "en-de": list(range(60, 100))}

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor([idx])}


class MockDatasetFromMetadata(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([idx]),
            "metadata": {
                "source_lang": "en",
                "target_lang": "fr" if idx < 60 else "de",
            },
        }


class PlainDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return torch.tensor([idx])


class TestTemperatureSampler:
    def test_init_default_temperature(self):
        dataset = MockDataset(size=100)
        sampler = TemperatureSampler(dataset, batch_size=8)
        assert sampler.temperature == 1.0
        assert sampler.batch_size == 8
        assert sampler.output_batches is True
        assert sampler.drop_last is False
        assert sorted(sampler.lang_pairs) == ["en-de", "en-fr"]

    def test_init_temperature_two(self):
        dataset = MockDataset(size=100)
        sampler = TemperatureSampler(dataset, batch_size=8, temperature=2.0)
        assert sampler.temperature == 2.0
        base = torch.tensor([60, 40], dtype=torch.float)
        scaled = torch.log(base) / 2.0
        expected = torch.softmax(scaled, dim=0)
        assert torch.allclose(sampler.sampling_weights, expected, atol=1e-6)

    def test_init_temperature_sharpens(self):
        dataset = MockDataset(size=100)
        sampler = TemperatureSampler(dataset, batch_size=8, temperature=0.5)
        base = torch.tensor([60, 40], dtype=torch.float)
        scaled = torch.log(base) / 0.5
        expected = torch.softmax(scaled, dim=0)
        assert torch.allclose(sampler.sampling_weights, expected, atol=1e-6)

    def test_init_temperature_one_proportional(self):
        dataset = MockDataset(size=100)
        sampler = TemperatureSampler(dataset, batch_size=8, temperature=1.0)
        expected = torch.tensor([60, 40], dtype=torch.float) / 100.0
        assert torch.allclose(sampler.sampling_weights, expected, atol=1e-6)

    def test_temperature_zero_produces_inf_weights(self):
        """Temperature == 0 causes division by zero -> inf weights, not an exception."""
        dataset = MockDataset(size=100)
        sampler = TemperatureSampler(dataset, batch_size=8, temperature=0.0)
        assert torch.isinf(sampler.sampling_weights).any() or torch.isnan(sampler.sampling_weights).any()

    def test_output_individual_len_and_type(self):
        dataset = MockDataset(size=100)
        sampler = TemperatureSampler(
            dataset, batch_size=8, temperature=1.0, output_batches=False,
        )
        assert len(sampler) == 100

    def test_output_batches(self):
        dataset = MockDataset(size=100)
        sampler = TemperatureSampler(
            dataset, batch_size=8, temperature=1.0, output_batches=True,
        )
        expected = (100 + 8 - 1) // 8
        assert len(sampler) == expected
        batches = list(sampler)
        assert len(batches) == expected
        for b in batches:
            assert len(b) == 8

    def test_drop_last(self):
        dataset = MockDataset(size=100)
        sampler = TemperatureSampler(
            dataset, batch_size=12, temperature=1.0, drop_last=True,
        )
        expected = 100 // 12
        assert len(sampler) == expected
        batches = list(sampler)
        assert len(batches) == expected
        for b in batches:
            assert len(b) == 12

    def test_get_lang_pair_indices_method(self):
        dataset = MockDataset(size=100)
        sampler = TemperatureSampler(dataset, batch_size=8)
        assert sampler.lang_pairs == ["en-de", "en-fr"] or sampler.lang_pairs == ["en-fr", "en-de"]
        assert len(sampler.lang_pair_indices["en-fr"]) == 60
        assert len(sampler.lang_pair_indices["en-de"]) == 40

    def test_get_lang_pair_indices_from_metadata(self):
        dataset = MockDatasetFromMetadata(size=100)
        sampler = TemperatureSampler(dataset, batch_size=8)
        assert "en-fr" in sampler.lang_pairs
        assert "en-de" in sampler.lang_pairs
        assert len(sampler.lang_pair_indices["en-fr"]) == 60
        assert len(sampler.lang_pair_indices["en-de"]) == 40

    def test_fallback_default_pair(self):
        dataset = PlainDataset()
        sampler = TemperatureSampler(dataset, batch_size=4, temperature=1.0)
        assert sampler.lang_pairs == ["default"]
        assert len(sampler.lang_pair_indices["default"]) == 10
