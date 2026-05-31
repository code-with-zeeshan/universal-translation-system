# utils/unified_validation.py
"""
Unified validation using Pydantic v2.
Provides InputValidator for backward compatibility with existing consumers.
"""

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum

from pydantic import BaseModel, ValidationError, create_model

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    INPUT = "input"
    CONFIG = "config"
    MODEL = "model"
    DATA = "data"
    PATH = "path"
    SYSTEM = "system"


class ValidationResult:
    def __init__(self, valid: bool = True):
        self.valid = valid
        self.errors: List[str] = []
        self.warnings: List[str] = []


class InputValidator:
    def validate_model(self, data: Any, model: Type[BaseModel]) -> BaseModel:
        if isinstance(data, BaseModel):
            return data
        if isinstance(data, dict):
            return model(**data)
        raise ValueError(f"Expected dict or BaseModel, got {type(data).__name__}")

    def validate_input(self, data: Dict[str, Any], rules: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()
        fields = {k: (type(v) if isinstance(v, type) else type(None), ...) for k, v in rules.items()}
        try:
            model = create_model("DynamicModel", **fields)
            model(**data)
        except ValidationError as e:
            result.valid = False
            result.errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        return result

    def sanitize_text(self, text: str, max_length: int = 10000) -> str:
        import html
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = text.strip()
        return text[:max_length]

    @staticmethod
    def validate_text_input(text: str, max_length: int = 5000) -> str:
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text).__name__}")
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty")
        return text[:max_length]

    @staticmethod
    def validate_language_code(code: str) -> bool:
        return bool(re.match(r'^[a-z]{2}(-[A-Z]{2})?$', code))


class ConfigValidator:
    def validate_config(self, config: Dict[str, Any], schema: Type[BaseModel]) -> BaseModel:
        if isinstance(config, BaseModel):
            return config
        return schema(**config)

    def validate_file_exists(self, path: str) -> bool:
        return Path(path).exists()

    def validate_path(self, path: str) -> bool:
        p = Path(path)
        resolved = p.resolve()
        return not any(part == ".." for part in p.parts) and resolved.exists()


class ModelValidator:
    def validate_model_path(self, path: str) -> bool:
        return Path(path).exists()

    def validate_model_architecture(self, model: Any) -> bool:
        return hasattr(model, "forward") and callable(getattr(model, "forward"))


class DataValidator:
    def validate_text(self, text: str, min_length: int = 1, max_length: int = 10000) -> bool:
        return isinstance(text, str) and min_length <= len(text) <= max_length

    def validate_language_code(self, code: str) -> bool:
        return bool(re.match(r'^[a-z]{2}(-[A-Z]{2})?$', code))


class PathValidator:
    def validate_path(self, path: str) -> bool:
        return Path(path).exists()

    def validate_path_safety(self, path: str) -> bool:
        return ".." not in path.split("/") and ".." not in path.split("\\")


class SystemValidator:
    def validate_python_version(self, min_version: str = "3.8") -> bool:
        import sys
        parts = tuple(int(x) for x in sys.version.split()[0].split("."))
        min_parts = tuple(int(x) for x in min_version.split("."))
        return parts >= min_parts

    def check_disk_space(self, path: str = ".", min_gb: float = 1.0) -> bool:
        try:
            stat = os.statvfs(path)
            free_gb = (stat.f_frsize * stat.f_bavail) / (1024 ** 3)
            return free_gb >= min_gb
        except Exception:
            return True
