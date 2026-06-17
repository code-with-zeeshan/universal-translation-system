import hashlib
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class PipelineStage(Enum):
    """Pipeline execution stages"""
    DOWNLOAD_EVAL = "download_evaluation"
    DOWNLOAD_TRAIN = "download_training"
    DIRECT_OPUS = "direct_opus"
    SAMPLE_FILTER = "sample_filter"
    AUGMENT = "augment"  # includes Wikipedia monolingual download
    DISTILL = "knowledge_distillation"
    CREATE_READY = "create_ready"
    VALIDATE = "validate"
    VOCABULARY = "vocabulary"
    COMET_QUALITY = "comet_quality"


STAGE_ORDER = [
    PipelineStage.DOWNLOAD_EVAL,
    PipelineStage.DOWNLOAD_TRAIN,
    PipelineStage.DIRECT_OPUS,
    PipelineStage.SAMPLE_FILTER,
    PipelineStage.AUGMENT,
    PipelineStage.DISTILL,
    PipelineStage.CREATE_READY,
    PipelineStage.COMET_QUALITY,   # filters train_final.txt/val_final.txt which CREATE_READY produces
    PipelineStage.VALIDATE,
    PipelineStage.VOCABULARY,
]


def _hash_config_section(section: object) -> str:
    """Fingerprint a config section to detect input changes."""
    raw = json.dumps(section, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class PipelineState:
    """Track pipeline execution state with sub-stage and fingerprint support."""
    completed_stages: Dict[str, bool]
    current_stage: Optional[PipelineStage]
    total_sentences: int = 0
    total_size_gb: float = 0.0
    error_count: int = 0
    pipeline_complete: bool = False
    stage_input_hashes: Dict[str, str] = field(default_factory=dict)
    sub_stage_progress: Dict[str, List[str]] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if all stages are complete"""
        return all(self.completed_stages.values())

    def get_progress(self) -> float:
        """Get overall progress percentage"""
        if not self.completed_stages:
            return 0.0
        completed = sum(1 for v in self.completed_stages.values() if v)
        return (completed / len(self.completed_stages)) * 100

    def mark_sub_stage(self, stage_name: str, item: str):
        """Record a sub-item as completed within a stage."""
        if stage_name not in self.sub_stage_progress:
            self.sub_stage_progress[stage_name] = []
        if item not in self.sub_stage_progress[stage_name]:
            self.sub_stage_progress[stage_name].append(item)

    def is_sub_stage_done(self, stage_name: str, item: str) -> bool:
        """Check if a sub-item was already completed."""
        return item in self.sub_stage_progress.get(stage_name, [])

    def invalidate_downstream(self, from_stage: PipelineStage):
        """Clear completion and sub-progress for from_stage and all later stages."""
        found = False
        for stage in STAGE_ORDER:
            if stage == from_stage:
                found = True
            if found:
                s = stage.value
                self.completed_stages[s] = False
                self.sub_stage_progress.pop(s, None)
                self.stage_input_hashes.pop(s, None)
        self.pipeline_complete = False

    def to_dict(self) -> dict:
        return {
            'completed_stages': self.completed_stages,
            'current_stage': self.current_stage.value if self.current_stage else None,
            'total_sentences': self.total_sentences,
            'total_size_gb': self.total_size_gb,
            'error_count': self.error_count,
            'pipeline_complete': self.pipeline_complete,
            'stage_input_hashes': self.stage_input_hashes,
            'sub_stage_progress': self.sub_stage_progress,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PipelineState':
        state = cls(
            completed_stages=data.get('completed_stages', {}),
            current_stage=PipelineStage(data['current_stage']) if data.get('current_stage') else None,
            total_sentences=data.get('total_sentences', 0),
            total_size_gb=data.get('total_size_gb', 0.0),
            error_count=data.get('error_count', 0),
            pipeline_complete=data.get('pipeline_complete', False),
            stage_input_hashes=data.get('stage_input_hashes', {}),
            sub_stage_progress=data.get('sub_stage_progress', {}),
        )
        return state
