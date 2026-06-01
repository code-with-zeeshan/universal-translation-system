from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional


class PipelineStage(Enum):
    """Pipeline execution stages"""
    DOWNLOAD_EVAL = "download_evaluation"
    DOWNLOAD_TRAIN = "download_training"
    WIKIPEDIA_BT = "wikipedia_backtranslation"
    DIRECT_OPUS = "direct_opus"
    SAMPLE_FILTER = "sample_filter"
    AUGMENT = "augment"
    DISTILL = "knowledge_distillation"
    CREATE_READY = "create_ready"
    VALIDATE = "validate"
    VOCABULARY = "vocabulary"
    COMET_QUALITY = "comet_quality"


@dataclass
class PipelineState:
    """Track pipeline execution state"""
    completed_stages: Dict[str, bool]
    current_stage: Optional[PipelineStage]
    total_sentences: int = 0
    total_size_gb: float = 0.0
    error_count: int = 0
    
    def is_complete(self) -> bool:
        """Check if all stages are complete"""
        return all(self.completed_stages.values())
    
    def get_progress(self) -> float:
        """Get overall progress percentage"""
        if not self.completed_stages:
            return 0.0
        completed = sum(1 for v in self.completed_stages.values() if v)
        return (completed / len(self.completed_stages)) * 100
