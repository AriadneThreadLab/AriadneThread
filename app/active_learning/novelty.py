"""Diversity control: keep the review queue varied instead of repetitive.

Three cheap, deterministic layers run on the request path:

1. a normalized query hash for exact repeats,
2. a task signature (``comparison|park|radius|count``) for structural repeats,
3. token-set overlap for near-duplicate wording.

A fourth, optional layer reuses the existing BGE-M3 infrastructure through
:class:`SemanticSimilarityIndex`. It is deliberately *not* wired into request
handling: embedding a query costs far more than the rest of this pipeline, so
semantic enrichment belongs in a CLI batch job.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.active_learning.contracts import TaskType

_NON_WORD = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "best",
        "both",
        "by",
        "choose",
        "compare",
        "for",
        "from",
        "in",
        "is",
        "me",
        "of",
        "on",
        "or",
        "return",
        "show",
        "the",
        "them",
        "to",
        "ween",
        "with",
        "within",
    }
)

#: Token-set overlap at or above this level counts as "the same question again".
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.85


@runtime_checkable
class SemanticSimilarityIndex(Protocol):
    """Optional embedding-backed similarity lookup (BGE-M3 reuse)."""

    async def max_similarity(self, text: str) -> float: ...


@dataclass(frozen=True, slots=True)
class NoveltyAssessment:
    """How new this run looks compared with what is already stored."""

    query_hash: str
    task_signature: str
    duplicate_count: int
    nearest_similarity: float
    is_novel: bool


def normalize_query(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace."""
    lowered = _NON_WORD.sub(" ", text.lower())
    return " ".join(lowered.split())


def query_fingerprint(text: str) -> str:
    """Stable hash of the normalized query, with numbers kept intact.

    Numbers matter here: a 2 km comparison and a 5 km comparison are different
    training examples even though the wording is identical.
    """
    return hashlib.sha256(normalize_query(text).encode("utf-8")).hexdigest()


def query_tokens(text: str) -> frozenset[str]:
    """Content tokens used for near-duplicate detection."""
    return frozenset(
        token
        for token in normalize_query(text).split()
        if token not in _STOP_WORDS and len(token) > 1
    )


def jaccard_similarity(left: str, right: str) -> float:
    """Token-set overlap in ``[0, 1]``."""
    left_tokens = query_tokens(left)
    right_tokens = query_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / float(len(union))


def is_near_duplicate(
    left: str,
    right: str,
    *,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> bool:
    """Whether two queries are the same question in different words.

    Distinct numbers (radii, limits) always break the match, because they
    change the expected plan.
    """
    if set(_NUMBER.findall(left)) != set(_NUMBER.findall(right)):
        return False
    return jaccard_similarity(left, right) >= threshold


def concept_token(concept: str | None) -> str:
    """Reduce a feature concept or OSM tag to one signature token."""
    if not concept:
        return "feature"
    text = concept.strip().lower()
    if "=" in text:
        text = text.split("=", 1)[1]
    words = [word for word in _NON_WORD.sub(" ", text).split() if word not in {"public", "osm"}]
    if not words:
        return "feature"
    token = words[-1]
    if len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def build_task_signature(
    *,
    task_type: TaskType,
    feature_concept: str | None,
    scope_kind: str,
    metric: str | None,
) -> str:
    """Structural fingerprint such as ``comparison|park|radius|count``."""
    parts = [task_type.value, concept_token(feature_concept), scope_kind or "none"]
    if metric:
        parts.append(metric)
    return "|".join(parts)
