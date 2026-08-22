"""Build the analyze_features tool and its pure collaborators."""

from __future__ import annotations

from app.analytics.comparison import ComparisonEngine
from app.analytics.engine import SpatialAnalyticsEngine
from app.analytics.feasibility import MetricFeasibilityValidator
from app.analytics.report import ReportAssembler
from app.tools.analyze_features import AnalyzeFeaturesTool


def build_analyze_features_tool() -> AnalyzeFeaturesTool:
    """Compose the analytics tool from stateless collaborators."""
    return AnalyzeFeaturesTool(
        engine=SpatialAnalyticsEngine(),
        validator=MetricFeasibilityValidator(),
        comparison=ComparisonEngine(),
        reporter=ReportAssembler(),
    )
