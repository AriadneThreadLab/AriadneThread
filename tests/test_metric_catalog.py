"""Metric catalog completeness and stability."""

from __future__ import annotations

from typing import get_args

from app.analytics.catalog import METRIC_CATALOG, METRIC_CATALOG_VERSION, get_metric_definition
from app.analytics.contracts import MetricType
from app.analytics.engine import SpatialAnalyticsEngine


def test_catalog_covers_every_metric_type() -> None:
    assert set(METRIC_CATALOG) == set(get_args(MetricType))


def test_catalog_version_is_stable() -> None:
    assert METRIC_CATALOG_VERSION == "metric-catalog-1"


def test_every_implementation_id_resolves() -> None:
    engine = SpatialAnalyticsEngine()
    for metric, definition in METRIC_CATALOG.items():
        assert definition.implementation_id.startswith("SpatialAnalyticsEngine.")
        method = definition.implementation_id.split(".", 1)[1]
        assert hasattr(engine, method), metric
        assert definition.unit
        assert definition.display_label


def test_count_allows_zero_observations() -> None:
    assert get_metric_definition("count").zero_observations_allowed is True
    assert get_metric_definition("mean").zero_observations_allowed is False
