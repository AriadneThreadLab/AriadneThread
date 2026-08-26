"""Indicator Catalog load, query, and validation (offline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import pytest
import yaml
from app.analytics.catalog import METRIC_CATALOG
from app.analytics.contracts import MetricType
from app.analytics.methods import METHOD_REGISTRY, MethodId, get_method_spec
from app.analytics.rules import STABLE_RULE_IDS
from app.core.errors import IndicatorCatalogError, UnknownIndicatorError
from app.indicators.catalog import (
    INDICATOR_CATALOG,
    INDICATOR_CATALOG_VERSION,
    AvailableCapabilities,
    find_candidates,
    get_indicator,
    list_by_domain,
    list_indicators,
    load_indicator_catalog,
)
from app.indicators.contracts import parse_tag_literal
from app.osm.query_spec import TagFilter

EXPECTED_IDS = (
    "feature_count",
    "green_accessibility",
    "green_diversity",
    "green_space_ratio",
    "intersection_density",
    "poi_density",
    "road_accessibility",
    "road_density",
    "service_accessibility",
    "service_diversity",
)


def test_catalog_loads_and_registers_phase1_indicators() -> None:
    ids = tuple(sorted(INDICATOR_CATALOG.indicator_ids()))
    assert ids == EXPECTED_IDS
    assert INDICATOR_CATALOG_VERSION == "indicator-catalog-1"
    assert len(list_indicators()) == 10


def test_feature_count_remains_available() -> None:
    definition = get_indicator("feature_count")
    assert definition.domain == "core"
    assert definition.method_id == "count"
    assert definition.status == "stable"
    spec = get_method_spec("count")
    assert spec.metric_primitives == ("count",)
    assert spec.implemented is True
    assert "count" in METRIC_CATALOG


def test_unknown_indicator_fails() -> None:
    with pytest.raises(UnknownIndicatorError, match="unknown indicator_id"):
        get_indicator("invented_green_score")


def test_domain_filtering() -> None:
    green = list_by_domain("green_space")
    assert {item.indicator_id for item in green} == {
        "green_space_ratio",
        "green_accessibility",
        "green_diversity",
    }
    mobility = list_by_domain("mobility")
    assert {item.indicator_id for item in mobility} == {
        "road_accessibility",
        "road_density",
        "intersection_density",
    }
    services = list_by_domain("urban_services")
    assert {item.indicator_id for item in services} == {
        "poi_density",
        "service_accessibility",
        "service_diversity",
    }
    core = list_by_domain("core")
    assert [item.indicator_id for item in core] == ["feature_count"]


def test_candidate_filtering_by_goal() -> None:
    coverage = find_candidates("green_space", "coverage")
    assert [item.indicator_id for item in coverage] == ["green_space_ratio"]
    access = find_candidates("green_space", "accessibility")
    assert [item.indicator_id for item in access] == ["green_accessibility"]
    none = find_candidates("green_space", "abundance")
    assert none == ()


def test_candidate_filtering_by_implemented_capabilities() -> None:
    implemented = find_candidates(
        "green_space",
        "coverage",
        AvailableCapabilities(implemented_only=True),
    )
    assert [item.indicator_id for item in implemented] == ["green_space_ratio"]

    diversity = find_candidates(
        "green_space",
        "variability",
        AvailableCapabilities(implemented_only=True),
    )
    assert [item.indicator_id for item in diversity] == ["green_diversity"]

    entropy_only = find_candidates(
        "green_space",
        "variability",
        AvailableCapabilities(method_ids=frozenset({"shannon_entropy"})),
    )
    assert [item.indicator_id for item in entropy_only] == ["green_diversity"]

    polygon_only = find_candidates(
        "mobility",
        "concentration",
        AvailableCapabilities(geometries=frozenset({"polygon"})),
    )
    assert polygon_only == ()


def test_indicator_binds_to_existing_metric_primitives() -> None:
    ratio = get_indicator("green_space_ratio")
    area_share = get_method_spec(ratio.method_id)
    assert area_share.metric_primitives == ("coverage_percentage",)
    assert "coverage_percentage" in METRIC_CATALOG

    poi = get_indicator("poi_density")
    assert get_method_spec(poi.method_id).metric_primitives == ("density",)

    road_access = get_indicator("road_accessibility")
    assert get_method_spec(road_access.method_id).metric_primitives == ("nearest_distance",)


def test_method_registry_covers_every_method_id() -> None:
    assert set(METHOD_REGISTRY) == set(get_args(MethodId))


def test_semantic_rules_are_unchanged() -> None:
    assert "ABUNDANCE_COUNT_001" in STABLE_RULE_IDS
    assert set(METRIC_CATALOG) == set(get_args(MetricType))


def test_parse_tag_literal_uses_osm_tag_filter() -> None:
    named = parse_tag_literal("leisure=park")
    assert named == TagFilter(key="leisure", value="park")
    key_only = parse_tag_literal("amenity")
    assert key_only == TagFilter(key="amenity", value=None)


def test_duplicate_ids_fail(tmp_path: Path) -> None:
    _write_minimal_count(tmp_path / "core" / "a.yaml", indicator_id="feature_count")
    _write_minimal_count(tmp_path / "core" / "b.yaml", indicator_id="feature_count")
    with pytest.raises(IndicatorCatalogError, match="duplicate indicator_id"):
        load_indicator_catalog(tmp_path)


def test_unknown_domain_in_definition_fails(tmp_path: Path) -> None:
    path = tmp_path / "core" / "bad.yaml"
    payload = _minimal_count_payload("feature_count")
    payload["domain"] = "hydrology"
    _write_yaml(path, payload)
    with pytest.raises(IndicatorCatalogError, match="invalid indicator definition"):
        load_indicator_catalog(tmp_path)


def test_unknown_method_id_fails(tmp_path: Path) -> None:
    path = tmp_path / "core" / "bad.yaml"
    payload = _minimal_count_payload("feature_count")
    payload["method_id"] = "invented_formula"
    _write_yaml(path, payload)
    with pytest.raises(IndicatorCatalogError, match="invalid indicator definition"):
        load_indicator_catalog(tmp_path)


def test_missing_feature_spec_fails(tmp_path: Path) -> None:
    path = tmp_path / "core" / "bad.yaml"
    payload = _minimal_count_payload("feature_count")
    payload["requirements"] = [
        {
            "requirement_id": "subject_features",
            "kind": "osm_features",
            "source": "openstreetmap",
        }
    ]
    _write_yaml(path, payload)
    with pytest.raises(IndicatorCatalogError, match="feature_spec"):
        load_indicator_catalog(tmp_path)


def test_malformed_tag_fails(tmp_path: Path) -> None:
    path = tmp_path / "green_space" / "bad.yaml"
    payload = _minimal_count_payload("green_space_ratio")
    payload["domain"] = "green_space"
    payload["method_id"] = "area_share"
    payload["goals"] = ["coverage"]
    payload["references"] = [{"citation": "A published methodology note."}]
    payload["requirements"][0]["feature_spec"]["categories"][0]["tags"] = ['leisure="park']
    _write_yaml(path, payload)
    with pytest.raises(IndicatorCatalogError, match="invalid indicator definition"):
        load_indicator_catalog(tmp_path)


def test_documentation_url_must_be_whitelisted(tmp_path: Path) -> None:
    path = tmp_path / "core" / "bad.yaml"
    payload = _minimal_count_payload("feature_count")
    payload["requirements"][0]["feature_spec"]["documentation"] = [
        {
            "title": "Not a whitelist page",
            "url": "https://wiki.openstreetmap.org/wiki/Tag:amenity%3Dfuel",
        }
    ]
    _write_yaml(path, payload)
    with pytest.raises(IndicatorCatalogError, match="whitelist"):
        load_indicator_catalog(tmp_path)


def test_nontrivial_method_requires_references(tmp_path: Path) -> None:
    path = tmp_path / "green_space" / "bad.yaml"
    payload = _minimal_count_payload("green_space_ratio")
    payload["domain"] = "green_space"
    payload["method_id"] = "area_share"
    payload["goals"] = ["coverage"]
    payload["references"] = []
    payload["requirements"][0]["feature_spec"]["categories"][0]["tags"] = ["leisure=park"]
    _write_yaml(path, payload)
    with pytest.raises(IndicatorCatalogError, match="must declare at least one reference"):
        load_indicator_catalog(tmp_path)


def test_invalid_parameter_fails(tmp_path: Path) -> None:
    path = tmp_path / "green_space" / "bad.yaml"
    payload = _minimal_count_payload("green_diversity")
    payload["domain"] = "green_space"
    payload["method_id"] = "shannon_entropy"
    payload["goals"] = ["variability"]
    payload["parameters"] = {"log_base": 1.0}
    payload["references"] = [{"citation": "Shannon 1948 communication theory paper."}]
    payload["requirements"][0]["feature_spec"]["categories"][0]["tags"] = ["leisure=park"]
    _write_yaml(path, payload)
    with pytest.raises(IndicatorCatalogError, match="greater than"):
        load_indicator_catalog(tmp_path)


def test_folder_must_match_domain(tmp_path: Path) -> None:
    path = tmp_path / "mobility" / "misplaced.yaml"
    _write_yaml(path, _minimal_count_payload("feature_count"))
    with pytest.raises(IndicatorCatalogError, match="lives under"):
        load_indicator_catalog(tmp_path)


def _minimal_count_payload(indicator_id: str) -> dict[str, Any]:
    return {
        "indicator_id": indicator_id,
        "domain": "core",
        "display_label": "Feature count",
        "purpose": "How many matching features exist in the analysis scope.",
        "goals": ["abundance"],
        "method_id": "count",
        "formula": "n(features)",
        "unit": "features",
        "display_precision": 0,
        "direction": "higher_is_better",
        "min_observations": 0,
        "status": "stable",
        "since_catalog_version": "indicator-catalog-1",
        "requirements": [
            {
                "requirement_id": "subject_features",
                "kind": "osm_features",
                "source": "openstreetmap",
                "feature_spec": {
                    "geometry": "any",
                    "element_types": ["node", "way", "relation"],
                    "categories": [
                        {"id": "subject", "label": "Subject features", "tags": []},
                    ],
                },
            }
        ],
        "limitations": ["osm_completeness"],
        "references": [{"citation": "Ariadne Thread metric catalog, metric-catalog-1 (count)."}],
    }


def _write_minimal_count(path: Path, *, indicator_id: str) -> None:
    _write_yaml(path, _minimal_count_payload(indicator_id))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
