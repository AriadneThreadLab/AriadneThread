"""SolarDSS UI pass-through: copied indicators, GeoJSON, and charts. No science."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from app.agent.accumulator import ResultAccumulator
from app.agent.contracts import GeoAgentResponse, SourceReference
from app.agent.evidence import partition_evidence
from app.api.schemas import to_agent_query_response
from app.solar.display import (
    SOLAR_MAP_LAYER_IDS,
    copy_solar_charts,
    copy_solar_geojson,
    copy_solar_outputs,
    copy_solar_support,
)
from app.tools.solar_tools import SOLAR_MAP_ATTRIBUTION, AnalyzeSolarSiteResult

_WEB = Path(__file__).resolve().parents[1] / "app" / "web"
_JS = _WEB / "static" / "js"


def _plugin_payload() -> dict[str, object]:
    return {
        "status": "success",
        "capability": "solar_candidate_ranking",
        "statistics": {
            "ghi": 1820.5,
            "dni": 2100.0,
            "slope": 4.2,
            "available_area": 12500.0,
            "distance_to_grid": 340.0,
            "distance_to_road": 80.0,
            "suitability_score": 0.81,
            "recommended_capacity": 250.0,
            "tilt": 30.0,
            "azimuth": 180.0,
            "annual_energy": 412000.0,
            "specific_yield": 1648.0,
            "protected_area_status": "outside",
            "module_technology": "mono-Si",
        },
        "selected_site": "A",
        "mode": "site_selection",
        "project_type": "ground_mounted",
        "candidates": [
            {
                "candidate_id": "A",
                "name": "North parcel",
                "rank": 1,
                "score": 0.81,
                "ghi": 1820.5,
                "slope": 4.2,
                "distance_to_grid": 340.0,
                "annual_energy": 412000.0,
                "selected": True,
            },
            {
                "candidate_id": "B",
                "name": "South parcel",
                "rank": 2,
                "score": 0.64,
                "ghi": 1710.0,
                "slope": 8.1,
                "distance_to_grid": 910.0,
                "annual_energy": 301000.0,
            },
        ],
        "geojson": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[16.3, 48.2], [16.4, 48.2], [16.4, 48.3], [16.3, 48.3], [16.3, 48.2]]
                        ],
                    },
                    "properties": {
                        "candidate_id": "A",
                        "solar_layer": "candidate_sites",
                        "score": 0.81,
                        "ghi": 1820.5,
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"candidate_id": "ghost", "solar_layer": "candidate_sites"},
                },
            ],
        },
        "charts": [
            {
                "chart_id": "monthly_pv_generation",
                "title": "Monthly PV generation",
                "chart_type": "bar",
                "x_label": "Month",
                "y_label": "kWh",
                "series": [
                    {
                        "name": "PV generation",
                        "data": [{"x": 1, "y": 28000.0}, {"x": 6, "y": 51000.0}],
                    }
                ],
                "metadata": {"engine": "SolarDSS", "capability_id": "solar_candidate_ranking"},
            },
            {
                "chart_id": "invented",
                "title": "LLM chart",
                "chart_type": "pie",
                "series": [{"name": "bad", "data": [{"x": "a", "y": 1}]}],
            },
            {
                "chart_id": "empty_series",
                "title": "Empty",
                "chart_type": "bar",
                "series": [],
            },
        ],
        "provenance": {
            "engine": "SolarDSS",
            "source": "integration_layer",
            "capability_id": "solar_candidate_ranking",
            "solardss_version": "0.1.0",
        },
    }


def test_copy_solar_support_copies_indicators_and_leaves_gaps() -> None:
    support = copy_solar_support(_plugin_payload())
    assert support is not None
    assert support.engine == "SolarDSS"
    assert support.ghi == 1820.5
    assert support.expected_annual_energy == 412000.0
    assert support.protected_area_status == "outside"
    assert support.selected_site == "A"
    assert [row.candidate_id for row in support.candidates] == ["A", "B"]
    assert support.candidates[0].selected is True
    sparse = copy_solar_support(
        {"statistics": {"ghi": 1500.0}, "provenance": {"engine": "SolarDSS"}}
    )
    assert sparse is not None
    assert sparse.ghi == 1500.0
    assert sparse.dni is None
    assert sparse.slope is None
    assert sparse.candidates == ()


def test_copy_solar_support_does_not_invent_ghi_from_other_fields() -> None:
    support = copy_solar_support(
        {
            "statistics": {"slope": 3.0, "annual_energy": 1000.0},
            "provenance": {"engine": "SolarDSS"},
        }
    )
    assert support is not None
    assert support.ghi is None
    assert support.specific_yield is None
    assert support.slope == 3.0


def test_copy_solar_geojson_drops_features_without_geometry() -> None:
    geojson = copy_solar_geojson(_plugin_payload()["geojson"])
    assert geojson is not None
    assert len(geojson["features"]) == 1
    feature = geojson["features"][0]
    assert feature["properties"]["candidate_id"] == "A"
    assert feature["geometry"]["type"] == "Polygon"
    assert copy_solar_geojson({"type": "FeatureCollection", "features": []}) is None
    assert copy_solar_geojson({"type": "Feature", "geometry": {"type": "Point"}}) is None


def test_copy_solar_charts_keeps_engine_series_and_drops_malformed() -> None:
    charts = copy_solar_charts(_plugin_payload()["charts"])
    assert [item.chart_id for item in charts] == ["monthly_pv_generation"]
    assert charts[0].series[0].data[0].y == 28000.0
    assert charts[0].metadata.engine == "SolarDSS"


def test_copy_solar_outputs_round_trip() -> None:
    support, geojson, charts = copy_solar_outputs(_plugin_payload())
    assert support is not None
    assert support.decision_mode == "site_selection"
    assert geojson is not None
    assert len(geojson["features"]) == 1
    assert len(charts) == 1


def test_accumulator_routing_only_does_not_invent_map_or_indicators() -> None:
    state = ResultAccumulator()
    state.absorb(
        "analyze_solar_site",
        AnalyzeSolarSiteResult(
            status="routed",
            capability="solar_site_selection",
            mode="site_selection",
            detail="routed",
        ),
    )
    assert state.geojson is None
    assert state.solar_results is None
    assert state.charts == []


def test_accumulator_copies_plugin_geojson_charts_and_support() -> None:
    support, geojson, charts = copy_solar_outputs(_plugin_payload())
    state = ResultAccumulator()
    state.absorb(
        "analyze_solar_site",
        AnalyzeSolarSiteResult(
            status="routed",
            capability="solar_candidate_ranking",
            mode="site_selection",
            project_type="ground_mounted",
            source_title="SolarDSS candidate ranking",
            geojson=geojson,
            charts=charts,
            support=support,
            feature_count=1,
        ),
    )
    assert state.geojson is not None
    assert len(state.geojson["features"]) == 1
    assert state.feature_count == 1
    assert state.solar_results is not None
    assert state.solar_results.ghi == 1820.5
    assert state.charts[0].chart_id == "monthly_pv_generation"
    assert any(source.kind == "solar_analysis" for source in state.sources)


def test_api_exposes_solar_results_and_map_attribution() -> None:
    support, geojson, charts = copy_solar_outputs(_plugin_payload())
    result = GeoAgentResponse(
        answer="Copied SolarDSS ranking.",
        sources=[SourceReference(kind="solar_analysis", title="SolarDSS candidate ranking")],
        geojson=geojson,
        feature_count=1,
        solar_results=support,
        charts=charts,
        stop_reason="final_answer",
        model="fake",
    )
    payload = to_agent_query_response(result, request_id="req-1")
    assert payload.solar_results is not None
    assert payload.solar_results.ghi == 1820.5
    assert payload.live_data_available is True
    assert payload.live_query_executed is True
    assert payload.geojson is not None
    assert SOLAR_MAP_ATTRIBUTION in (payload.attribution or "")
    dumped = json.loads(payload.model_dump_json())
    assert dumped["solar_results"]["ghi"] == 1820.5
    assert dumped["charts"][0]["series"][0]["data"][0]["y"] == 28000.0


def test_evidence_partition_keeps_copied_solardss_fields() -> None:
    support, _, _ = copy_solar_outputs(_plugin_payload())
    evidence = partition_evidence(solar_results=support)
    assert len(evidence.deterministic_analytical) == 1
    item = evidence.deterministic_analytical[0]
    assert item.engine == "SolarDSS"
    assert item.payload["science"] == "copied_from_solardss"
    assert "ghi" in item.payload["copied_fields"]
    assert evidence.knowledge_documents == ()


def test_ui_chrome_includes_solar_panel_and_layers() -> None:
    html = (_WEB / "templates" / "index.html").read_text(encoding="utf-8")
    main_js = (_JS / "main.js").read_text(encoding="utf-8")
    map_js = (_JS / "map.js").read_text(encoding="utf-8")
    panel_js = (_JS / "solar-panel.js").read_text(encoding="utf-8")
    assert "Solar Decision Support Results" in html
    assert 'id="solar-results-section"' in html
    assert 'id="map-solar-legend"' in html
    assert "renderSolarDecisionSupport" in main_js
    assert "selectSolarCandidate" in main_js
    assert "onSolarCandidateSelect" in main_js
    for layer_id in SOLAR_MAP_LAYER_IDS:
        assert f'"{layer_id}"' in map_js
    assert "Decision Mode" in panel_js
    assert "Candidate comparison" in panel_js
    assert "SolarDSS provenance" in panel_js
    assert "onSolarCandidateSelect" in map_js


def test_solar_panel_renderer_copies_values_and_hides_when_empty() -> None:
    panel_js = (_JS / "solar-panel.js").read_text(encoding="utf-8")
    dom_js = (_JS / "dom.js").read_text(encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "dom.mjs").write_text(dom_js, encoding="utf-8")
            (tmp_path / "solar-panel.mjs").write_text(
                panel_js.replace("./dom.js", "./dom.mjs"),
                encoding="utf-8",
            )
            probe = tmp_path / "probe.mjs"
            probe.write_text(
                """
import { renderSolarDecisionSupport, selectSolarCandidate } from "./solar-panel.mjs";

function makeNode(name) {
  const node = {
    tagName: String(name).toUpperCase(),
    className: "",
    id: "",
    hidden: false,
    dataset: {},
    children: [],
    style: {},
    textContent: "",
    firstChild: null,
    appendChild(child) {
      this.children.push(child);
      this.firstChild = this.children[0];
      return child;
    },
    removeChild(child) {
      this.children = this.children.filter((item) => item !== child);
      this.firstChild = this.children[0] || null;
      return child;
    },
    querySelector(selector) {
      const all = this.querySelectorAll(selector);
      return all[0] || null;
    },
    querySelectorAll(selector) {
      const out = [];
      const match = (item) => {
        if (selector.startsWith(".")) {
          return String(item.className || "").split(/\\s+/).includes(selector.slice(1));
        }
        if (selector.startsWith("#")) return item.id === selector.slice(1);
        if (selector === "tbody tr") {
          return item.tagName === "TR" && item.parent && item.parent.tagName === "TBODY";
        }
        if (selector === "tbody") return item.tagName === "TBODY";
        return item.tagName === String(selector).toUpperCase();
      };
      const walk = (item) => {
        if (match(item)) out.push(item);
        for (const child of item.children || []) {
          child.parent = item;
          walk(child);
        }
      };
      walk(this);
      return out;
    },
    addEventListener() {},
  };
  node.classList = {
    toggle(name, force) {
      const parts = new Set(String(node.className || "").split(/\\s+/).filter(Boolean));
      if (force) parts.add(name);
      else parts.delete(name);
      node.className = [...parts].join(" ");
    },
  };
  return node;
}

globalThis.document = {
  createElement(name) {
    return makeNode(name);
  },
};

function collectText(node) {
  const parts = [];
  const walk = (item) => {
    if (item && item.textContent) parts.push(String(item.textContent));
    for (const child of item.children || []) walk(child);
  };
  walk(node);
  return parts.join(" ");
}

const root = makeNode("div");
const section = { hidden: true };
const empty = renderSolarDecisionSupport(null, root, section);
if (empty !== 0 || section.hidden !== true) throw new Error("empty report should hide");

const report = {
  decision_mode: "site_selection",
  project_type: "ground_mounted",
  selected_site: "A",
  suitability_score: 0.81,
  ghi: 1820.5,
  dni: null,
  slope: 4.2,
  candidates: [
    {
      candidate_id: "A",
      name: "North parcel",
      score: 0.81,
      ghi: 1820.5,
      slope: 4.2,
      distance_to_grid: 340,
      annual_energy: 412000,
    },
    {
      candidate_id: "B",
      name: "South parcel",
      score: 0.64,
      ghi: 1710,
      slope: 8.1,
      distance_to_grid: 910,
      annual_energy: 301000,
    },
  ],
  provenance: { engine: "SolarDSS", source: "integration_layer" },
  units: { ghi: "kWh/m2" },
};
const shown = renderSolarDecisionSupport(report, root, section);
if (shown !== 1 || section.hidden) throw new Error("filled report hidden");
const blob = collectText(root);
if (!blob.includes("Decision Mode")) throw new Error("decision mode missing");
if (!blob.includes("site_selection")) throw new Error("mode value missing");
if (!blob.includes("1820.5 kWh/m2")) throw new Error("copied GHI missing");
if (!blob.includes("DNI")) throw new Error("DNI label missing");
if (!blob.includes("—")) throw new Error("missing DNI should be an em dash");
if (!blob.includes("North parcel")) throw new Error("comparison table missing");
if (!blob.includes("SolarDSS provenance")) throw new Error("provenance missing");
selectSolarCandidate("B");
const inspected = root.querySelector("#solar-inspected-body");
const inspectedBlob = collectText(inspected);
if (!inspectedBlob.includes("South parcel")) throw new Error("map selection did not inspect B");
if (inspectedBlob.includes("1820.5")) throw new Error("inspected B reused site A GHI");
""",
                encoding="utf-8",
            )
            subprocess.run(["node", str(probe)], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        assert "renderSolarDecisionSupport" in panel_js
    except subprocess.CalledProcessError as exc:
        raise AssertionError(exc.stderr or exc.stdout) from exc


def test_known_solar_layers_are_complete() -> None:
    assert SOLAR_MAP_LAYER_IDS == (
        "search_area",
        "excluded_areas",
        "candidate_sites",
        "top_ranked_sites",
        "solar_resource",
        "land_cover",
        "protected_areas",
        "roads",
        "power_lines",
        "substations",
        "buildings",
        "selected_site",
    )
