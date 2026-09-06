"""Generic analysis-chart contract. Values are copied, never invented."""

from __future__ import annotations

from app.analytics.charts import bar_chart, line_chart, parse_charts
from app.tools.energy_report import LAYER_PCA, build_energy_analysis_report


def test_parse_charts_keeps_one_valid_bar_chart() -> None:
    charts = parse_charts(
        [
            {
                "chart_id": "pca_explained_variance",
                "title": "PCA Explained Variance",
                "chart_type": "bar",
                "x_label": "Principal component",
                "y_label": "Explained variance",
                "series": [
                    {
                        "name": "Explained variance",
                        "data": [
                            {"x": "PC1", "y": 0.61},
                            {"x": "PC2", "y": 0.27},
                        ],
                    }
                ],
                "metadata": {"engine": "GeoLoadST", "capability_id": "multidim_pca_clustering"},
            }
        ]
    )
    assert len(charts) == 1
    assert charts[0].chart_type == "bar"
    assert [point.y for point in charts[0].series[0].data] == [0.61, 0.27]


def test_parse_charts_keeps_line_and_rejects_malformed() -> None:
    charts = parse_charts(
        [
            {
                "chart_id": "moran_time",
                "title": "Moran I over time",
                "chart_type": "line",
                "series": [{"name": "I", "data": [{"x": 1, "y": 0.2}, {"x": 2, "y": 0.3}]}],
            },
            {"chart_id": "bad", "title": "Missing series", "chart_type": "bar"},
            {"chart_id": "unknown", "title": "Pie", "chart_type": "pie", "series": []},
            "not-an-object",
            {
                "chart_id": "nan",
                "title": "NaN",
                "chart_type": "scatter",
                "series": [{"name": "x", "data": [{"x": "a", "y": float("nan")}]}],
            },
        ]
    )
    assert [item.chart_id for item in charts] == ["moran_time"]
    assert charts[0].chart_type == "line"


def test_parse_charts_empty_when_absent() -> None:
    assert parse_charts(None) == []
    assert parse_charts([]) == []
    assert parse_charts({}) == []


def test_helpers_omit_empty_point_lists() -> None:
    assert (
        bar_chart(
            chart_id="empty",
            title="Empty",
            points=[],
            x_label="x",
            y_label="y",
            series_name="s",
        )
        is None
    )
    chart = line_chart(
        chart_id="ok",
        title="Line",
        points=[(1, 0.5)],
        x_label="t",
        y_label="v",
        series_name="series",
    )
    assert chart is not None
    assert chart.series[0].data[0].y == 0.5


def test_pca_report_counts_copied_cluster_ids() -> None:
    report = build_energy_analysis_report(
        capability_id="multidim_pca_clustering",
        network_id="1-MV-urban--0-sw",
        statistics={"cumulative_variance": 0.9},
        features=[
            {"properties": {"cluster_id": 0, "bus_id": 1}},
            {"properties": {"cluster_id": 0, "bus_id": 2}},
            {"properties": {"cluster_id": 1, "bus_id": 3}},
        ],
    )
    assert report.kind == "multidim_pca_clustering"
    assert report.layer_name == LAYER_PCA
    assert report.cluster_counts == {"0": 2, "1": 1}
    assert report.statistic_items[0].value == 0.9
