"""
Tests for data.pipeline_state - pipeline execution state tracking.
"""

import json
import dataclasses
import pytest
from pipeline.data.state import PipelineStage, PipelineState


class TestPipelineStage:
    def test_enum_values(self):
        assert PipelineStage.DOWNLOAD_EVAL.value == "download_evaluation"
        assert PipelineStage.DOWNLOAD_TRAIN.value == "download_training"
        assert PipelineStage.SAMPLE_FILTER.value == "sample_filter"
        assert PipelineStage.AUGMENT.value == "augment"
        assert PipelineStage.CREATE_READY.value == "create_ready"
        assert PipelineStage.VALIDATE.value == "validate"
        assert PipelineStage.VOCABULARY.value == "vocabulary"

    def test_enum_members(self):
        members = {m.name for m in PipelineStage}
        assert members == {
            "DOWNLOAD_EVAL", "DOWNLOAD_TRAIN", "SAMPLE_FILTER",
            "AUGMENT", "CREATE_READY", "VALIDATE", "VOCABULARY",
        }

    def test_enum_length(self):
        assert len(PipelineStage) == 7


class TestPipelineState:
    def test_create_all_completed(self):
        state = PipelineState(
            completed_stages={s.value: True for s in PipelineStage},
            current_stage=None,
        )
        assert state.is_complete() is True
        assert state.total_sentences == 0
        assert state.total_size_gb == 0.0
        assert state.error_count == 0

    def test_create_incomplete(self):
        completed = {
            "download_evaluation": True,
            "download_training": True,
            "sample_filter": False,
            "augment": False,
            "create_ready": False,
            "validate": False,
            "vocabulary": False,
        }
        state = PipelineState(
            completed_stages=completed,
            current_stage=PipelineStage.SAMPLE_FILTER,
            total_sentences=50000,
            total_size_gb=2.5,
            error_count=1,
        )
        assert state.is_complete() is False
        assert state.current_stage == PipelineStage.SAMPLE_FILTER
        assert state.total_sentences == 50000
        assert state.total_size_gb == 2.5
        assert state.error_count == 1

    def test_is_complete_all_true(self):
        state = PipelineState(
            completed_stages={"a": True, "b": True, "c": True},
            current_stage=None,
        )
        assert state.is_complete() is True

    def test_is_complete_one_false(self):
        state = PipelineState(
            completed_stages={"a": True, "b": False},
            current_stage=None,
        )
        assert state.is_complete() is False

    def test_is_complete_empty(self):
        state = PipelineState(completed_stages={}, current_stage=None)
        assert state.is_complete() is True

    def test_get_progress_100(self):
        state = PipelineState(
            completed_stages={"a": True, "b": True, "c": True},
            current_stage=None,
        )
        assert state.get_progress() == 100.0

    def test_get_progress_50(self):
        state = PipelineState(
            completed_stages={"a": True, "b": False, "c": True, "d": False},
            current_stage=None,
        )
        assert state.get_progress() == 50.0

    def test_get_progress_empty(self):
        state = PipelineState(completed_stages={}, current_stage=None)
        assert state.get_progress() == 0.0

    def test_get_progress_zero(self):
        state = PipelineState(
            completed_stages={"a": False, "b": False},
            current_stage=None,
        )
        assert state.get_progress() == 0.0

    def test_serialize_to_dict(self):
        state = PipelineState(
            completed_stages={"download": True, "process": False},
            current_stage=PipelineStage.SAMPLE_FILTER,
            total_sentences=1000,
            total_size_gb=0.5,
            error_count=2,
        )
        d = dataclasses.asdict(state)
        assert d["completed_stages"] == {"download": True, "process": False}
        assert d["current_stage"] == PipelineStage.SAMPLE_FILTER
        assert d["total_sentences"] == 1000
        assert d["total_size_gb"] == 0.5
        assert d["error_count"] == 2

    def test_serialize_to_json(self):
        state = PipelineState(
            completed_stages={"stage1": True, "stage2": False},
            current_stage=PipelineStage.VALIDATE,
            total_sentences=500,
            total_size_gb=1.2,
            error_count=0,
        )
        d = dataclasses.asdict(state)
        d["current_stage"] = d["current_stage"].value if d["current_stage"] else None
        js = json.dumps(d)
        loaded = json.loads(js)
        assert loaded["completed_stages"] == {"stage1": True, "stage2": False}
        assert loaded["current_stage"] == "validate"
        assert loaded["total_sentences"] == 500

    def test_deserialize_from_dict(self):
        data = {
            "completed_stages": {"a": True, "b": False},
            "current_stage": "augment",
            "total_sentences": 2000,
            "total_size_gb": 3.0,
            "error_count": 3,
        }
        state = PipelineState(
            completed_stages=data["completed_stages"],
            current_stage=PipelineStage(data["current_stage"]),
            total_sentences=data["total_sentences"],
            total_size_gb=data["total_size_gb"],
            error_count=data["error_count"],
        )
        assert state.completed_stages == {"a": True, "b": False}
        assert state.current_stage == PipelineStage.AUGMENT
        assert state.total_sentences == 2000

    def test_deserialize_none_stage(self):
        data = {
            "completed_stages": {"a": True},
            "current_stage": None,
            "total_sentences": 0,
            "total_size_gb": 0.0,
            "error_count": 0,
        }
        state = PipelineState(**data)
        assert state.current_stage is None
        assert state.is_complete() is True

    def test_default_values(self):
        state = PipelineState(completed_stages={}, current_stage=None)
        assert state.total_sentences == 0
        assert state.total_size_gb == 0.0
        assert state.error_count == 0
