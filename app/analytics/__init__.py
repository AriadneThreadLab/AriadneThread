"""Bounded spatial analytics, metric selection and analytical traceability.

The LLM selects a validated AnalysisPlan; deterministic code in this package
computes every number. Structured provenance records how the executable
analytical decision was made — never chain-of-thought.
"""

from __future__ import annotations

from app.analytics.contracts import (
    AnalysisBlock,
    AnalysisPlan,
    AnalysisResult,
    ComparisonResult,
    MetricResult,
    MetricType,
)

__all__ = [
    "AnalysisBlock",
    "AnalysisPlan",
    "AnalysisResult",
    "ComparisonResult",
    "MetricResult",
    "MetricType",
]
