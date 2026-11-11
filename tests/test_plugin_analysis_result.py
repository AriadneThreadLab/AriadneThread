"""Unified scientific-plugin envelope. Values are copied, never invented."""

from __future__ import annotations

from importlib.util import find_spec

import pytest
from app.agent.accumulator import ResultAccumulator
from app.agent.contracts import GeoAgentResponse, SourceReference
from app.analytics.charts import parse_charts, radar_chart
from app.analytics.plugin_result import (
    scientific_result_from_energy,
    scientific_result_from_solar,
)
from app.api.schemas import to_agent_query_response
from app.solar.display import copy_solar_outputs
from app.tools.energy_report import build_energy_analysis_report
from app.tools.solar_tools import AnalyzeSolarSiteResult


def test_radar_chart_helper_and_parser() -> None:
    chart = radar_chart(
        chart_id="criteria",
        title="Breakdown",
        points=[("GHI", 0.8), ("slope", 0.4), ("grid", 0.6)],
        x_label="Criterion",
        y_label="Score",
        series_name="Copied",
    )
    assert chart is not None
    assert chart.chart_type == "radar"
    parsed = parse_charts([chart.model_dump(), {"chart_id": "bad", "chart_type": "pie"}])
    assert [item.chart_id for item in parsed] == ["criteria"]


def test_energy_envelope_copies_statistics_and_layer() -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [16.3, 48.2]},
            "properties": {"bus_id": "bus_1", "degree_centrality": 0.4},
        }
    ]
    report = build_energy_analysis_report(
        capability_id="topology_centrality",
        network_id="1-MV-urban--0-sw",
        statistics={"analyzed_bus_count": 1.0},
        features=features,
    )
    envelope = scientific_result_from_energy(
        report,
        geojson={"type": "FeatureCollection", "features": features},
        charts=[],
    )
    assert envelope is not None
    assert envelope.provenance.engine == "GeoLoadST"
    assert envelope.provenance.plugin_id == "ariadne_geoloadst"
    assert envelope.layers[0].id
    assert any(item.id == "analyzed_bus_count" for item in envelope.metrics)


def test_solar_envelope_copies_indicators_and_leaves_missing() -> None:
    support, geojson, charts = copy_solar_outputs(
        {
            "statistics": {
                "ghi": 1450.0,
                "aspect": 180.0,
                "shading_impact": 0.12,
                "annual_energy": 9000.0,
            },
            "selected_site": "A",
            "geojson": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [16.37, 48.2]},
                        "properties": {
                            "candidate_id": "A",
                            "analysis_layer": "candidate_sites",
                        },
                    }
                ],
            },
            "charts": [
                {
                    "chart_id": "monthly_pv_generation",
                    "title": "Monthly PV",
                    "chart_type": "line",
                    "series": [{"name": "E", "data": [{"x": 1, "y": 100.0}]}],
                    "metadata": {"engine": "SolarDSS"},
                }
            ],
            "provenance": {"engine": "SolarDSS", "plugin_id": "ariadne_solar"},
        }
    )
    envelope = scientific_result_from_solar(support, geojson=geojson, charts=charts)
    assert envelope is not None
    by_id = {item.id: item for item in envelope.metrics}
    assert by_id["ghi"].value == 1450.0
    assert by_id["aspect"].value == 180.0
    assert by_id["shading_impact"].value == 0.12
    assert by_id["slope"].status == "missing"
    assert by_id["slope"].value is None
    assert envelope.layers[0].id == "candidate_sites"
    assert envelope.recommendations
    assert envelope.provenance.engine == "SolarDSS"


def test_accumulator_exposes_agent_questions_and_analysis_result() -> None:
    support, geojson, charts = copy_solar_outputs(
        {
            "statistics": {"ghi": 1200.0},
            "geojson": geojson_point(),
            "provenance": {"engine": "SolarDSS"},
        }
    )
    state = ResultAccumulator()
    state.absorb(
        "analyze_solar_site",
        AnalyzeSolarSiteResult(
            status="needs_input",
            questions=[
                {
                    "field": "location",
                    "prompt": "Do you already have a fixed site?",
                    "required": True,
                }
            ],
            geojson=geojson,
            charts=charts,
            support=support,
        ),
    )
    assert state.agent_questions
    assert state.agent_questions[0].field == "location"
    assert state.analysis_result is not None
    payload = to_agent_query_response(
        GeoAgentResponse(
            answer="Need location.",
            sources=[SourceReference(kind="solar_analysis", title="SolarDSS")],
            geojson=state.geojson,
            solar_decision=state.solar_decision,
            solar_results=state.solar_results,
            charts=list(state.charts),
            analysis_result=state.analysis_result,
            agent_questions=list(state.agent_questions),
            stop_reason="final_answer",
            model="fake",
        ),
        request_id="q1",
    )
    assert payload.agent_questions[0].prompt.startswith("Do you already")
    assert payload.analysis_result is not None
    assert payload.analysis_result.metrics


def geojson_point() -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [16.3, 48.2]},
                "properties": {"candidate_id": "A", "solar_layer": "candidate_sites"},
            }
        ],
    }


@pytest.mark.skipif(find_spec("ariadne_solar") is None, reason="ariadne_solar not installed")
def test_solar_plugin_adapter_is_importable() -> None:
    from app.tools.solar_tools import PluginSolarAnalysisBackend, _require_solar_plugin

    module = _require_solar_plugin()
    assert callable(getattr(module, "analyze", None))
    assert PluginSolarAnalysisBackend() is not None
