"""Structured GeoLoadST reports copy engine values; they do not invent metrics."""

from __future__ import annotations

from app.tools.energy_report import (
    LAYER_TOPOLOGY,
    build_energy_analysis_report,
    stamp_ranked_entities,
)


def test_topology_report_summarises_copied_centrality_values() -> None:
    features = [
        {
            "type": "Feature",
            "properties": {
                "bus_id": 10,
                "degree_centrality": 0.2,
                "betweenness_centrality": 0.9,
                "closeness_centrality": 0.4,
            },
        },
        {
            "type": "Feature",
            "properties": {
                "bus_id": 11,
                "degree_centrality": 0.8,
                "betweenness_centrality": 0.1,
                "closeness_centrality": 0.3,
            },
        },
    ]
    report = build_energy_analysis_report(
        capability_id="topology_centrality",
        network_id="1-MV-urban--0-sw",
        statistics={},
        features=features,
    )
    assert report.analysis_name == "Topology centrality"
    assert report.network_id == "1-MV-urban--0-sw"
    assert report.entity_count == 2
    assert report.layer_name == LAYER_TOPOLOGY
    assert report.ranked_by == "betweenness_centrality"
    assert [item.entity_id for item in report.ranked_entities] == ["10", "11"]
    by_id = {item.metric_id: item for item in report.metric_summaries}
    assert by_id["degree_centrality"].maximum == 0.8
    assert by_id["degree_centrality"].mean == 0.5
    assert by_id["betweenness_centrality"].top_entity_id == "10"
    stamp_ranked_entities(features, report)
    first_props = features[0]["properties"]
    assert isinstance(first_props, dict)
    assert first_props["in_top_n"] is True
    assert first_props["rank"] == 1


def test_topology_report_omits_missing_metrics() -> None:
    report = build_energy_analysis_report(
        capability_id="topology_centrality",
        network_id="1-MV-urban--0-sw",
        statistics={},
        features=[
            {"type": "Feature", "properties": {"bus_id": 3, "degree_centrality": 0.4}},
            {"type": "Feature", "properties": {"bus_id": 4, "value": 0.1}},
        ],
    )
    assert [item.metric_id for item in report.metric_summaries] == ["degree_centrality"]
    assert report.columns == ["degree_centrality"]
    assert report.ranked_by == "degree_centrality"
    assert "betweenness_centrality" not in report.columns


def test_lisa_report_copies_moran_and_cluster_counts() -> None:
    report = build_energy_analysis_report(
        capability_id="lisa_instability",
        network_id="1-MV-urban--0-sw",
        statistics={"moran_i": 0.34, "p_value": 0.02},
        features=[
            {"properties": {"cluster_type": "HIGH_HIGH"}},
            {"properties": {"cluster_type": "HIGH_HIGH"}},
            {"properties": {"cluster_type": "LOW_LOW"}},
        ],
    )
    assert report.kind == "moran_lisa"
    assert report.statistic_items[0].key == "moran_i"
    assert report.cluster_counts == {"HIGH_HIGH": 2, "LOW_LOW": 1}
    assert report.entity_count == 3


def test_variogram_report_copies_range_statistics() -> None:
    report = build_energy_analysis_report(
        capability_id="space_time_variogram",
        network_id="1-MV-urban--0-sw",
        statistics={
            "space_range": 120.5,
            "time_range_steps": 6.0,
            "time_range_hours": 1.5,
            "analyzed_bus_count": 134.0,
            "max_valid_space_lag": 300.0,
            "max_valid_time_lag": 12.0,
        },
        features=[{"type": "Feature"}, {"type": "Feature"}],
    )
    assert report.kind == "space_time_variogram"
    assert report.analysis_name == "Space-time variogram"
    by_key = {item.key: item for item in report.statistic_items}
    assert by_key["space_range"].label == "Spatial range"
    assert by_key["space_range"].value == 120.5
    assert by_key["time_range_steps"].label == "Temporal range"
    assert by_key["time_range_hours"].label == "Temporal range (hours)"
    assert by_key["analyzed_bus_count"].label == "Number of analyzed buses"
    assert by_key["max_valid_space_lag"].label == "Valid spatial lag range"
    assert by_key["max_valid_time_lag"].label == "Valid temporal lag range"
