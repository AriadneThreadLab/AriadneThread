"""Privacy boundary between runtime interactions and exportable training data.

Two separate guarantees:

* Hidden reasoning is *rejected*, never stored. A candidate that would carry
  chain-of-thought markers is refused outright rather than cleaned up, so the
  failure is visible instead of silently half-applied.
* Credentials and other secrets are *redacted* in place, because a user request
  may legitimately mention a token-shaped string that should not end up in a
  training corpus.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.errors import SanitizationError

#: Markers that indicate model chain-of-thought. Stripped at the provider
#: boundary already; this is the last line of defence before persistence.
_HIDDEN_REASONING_RE = re.compile(
    r"(</?think\b|\breasoning_content\b|<\|(?:thought|reasoning)\|>)",
    re.IGNORECASE,
)

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{16,}\b"), "[redacted:api_key]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer [redacted:token]"),
    (re.compile(r"\b(?:ghp|gho|hf)_[A-Za-z0-9]{16,}\b"), "[redacted:token]"),
    (re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s/@]+:[^\s/@]+@\S+"), "[redacted:credentialed_url]"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|password|passwd|secret|access[_-]?token|token|authorization)\b"
            r"\s*[:=]\s*\S+"
        ),
        r"\1=[redacted]",
    ),
)

#: Depth bound so a malformed payload cannot cause unbounded recursion.
_MAX_DEPTH = 6


def contains_hidden_reasoning(text: str) -> bool:
    """Whether ``text`` carries chain-of-thought markers."""
    return _HIDDEN_REASONING_RE.search(text) is not None


def redact_secrets(text: str) -> str:
    """Replace credential-shaped substrings with stable redaction markers."""
    cleaned = text
    for pattern, replacement in _SECRET_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def sanitize_text(text: str, *, field: str) -> str:
    """Reject hidden reasoning, then redact secrets."""
    if contains_hidden_reasoning(text):
        raise SanitizationError(f"{field} must not contain hidden model reasoning")
    return redact_secrets(text)


def sanitize_payload(value: Any, *, field: str, _depth: int = 0) -> Any:
    """Recursively sanitize a JSON-compatible payload (plans, summaries)."""
    if _depth > _MAX_DEPTH:
        raise SanitizationError(f"{field} is nested too deeply to sanitize safely")
    if isinstance(value, str):
        return sanitize_text(value, field=field)
    if isinstance(value, dict):
        return {
            sanitize_text(str(key), field=field): sanitize_payload(
                item, field=field, _depth=_depth + 1
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_payload(item, field=field, _depth=_depth + 1) for item in value]
    if isinstance(value, bool | int | float) or value is None:
        return value
    # Unknown object types are not exportable training data.
    raise SanitizationError(f"{field} contains an unsupported value type")
