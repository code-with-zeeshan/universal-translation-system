from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class TranslationPair:
    """Single translation pair for evaluation"""
    source: str
    target: str
    source_lang: str
    target_lang: str
    predicted: Optional[str] = None
