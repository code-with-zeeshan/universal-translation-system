"""
utils/sensitive_filter.py
A minimal sensitive data filter that masks common PII patterns.
This is intentionally lightweight and safe; can be extended later.
"""
from __future__ import annotations
import re
from typing import Pattern

class SensitiveDataFilter:
    """Mask common sensitive data in logs/payloads without changing structure."""

    def __init__(self):
        # Precompile patterns for performance
        # - Emails
        # - Simple phone numbers
        # - Generic API keys/tokens (basic heuristic)
        self._patterns: list[tuple[Pattern[str], str]] = [
            # Emails
            (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<redacted_email>"),
            # Phone numbers (intl + US/local formats)
            (re.compile(r"\b(?:\+?\d[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b"), "<redacted_phone>"),
            # Generic API keys/tokens via key-like prefixes
            (re.compile(r"\b(?:api|secret|token|key|password|pwd|pass|bearer)[=_:\s\"]+[A-Za-z0-9-_]{8,}\b", re.IGNORECASE), "<redacted_secret>"),
            # JWT-like tokens (header.payload.signature)
            (re.compile(r"\beyJ[\w-]+\.[\w-]+\.[\w-]+\b"), "<redacted_jwt>"),
            # AWS Access Key IDs (AKIA/ASIA + 16 alnum)
            (re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"), "<redacted_aws_access_key>"),
            # AWS Secret Access Key (40 base64-like)
            (re.compile(r"\b[A-Za-z0-9/+=]{40}\b"), "<redacted_secret>"),
            # Credit card (simple Luhn-ish shapes, non-exhaustive)
            (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "<redacted_card>"),
        ]

    def sanitize(self, text: str) -> str:
        """Return text with sensitive patterns masked."""
        if not text:
            return text
        out = text
        for pat, repl in self._patterns:
            out = pat.sub(repl, out)
        return out