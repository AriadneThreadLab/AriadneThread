"""Offline checks for the English Ariadne Thread web UI."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from app.agent.contracts import GeoAgentResponse, TraceEvent
from app.bootstrap import ApplicationServices
from app.core.config import Settings
from app.main import create_app

_WEB_ROOT = Path(__file__).resolve().parents[1] / "app" / "web"
_INDEX = _WEB_ROOT / "templates" / "index.html"
_JS_DIR = _WEB_ROOT / "static" / "js"
_CSS = _WEB_ROOT / "static" / "css" / "app.css"

# Persian / Arabic script should not appear in UI chrome (labels, headings, buttons).
_PERSIAN_UI_RE = re.compile(r"[\u0600-\u06FF]")
_LEGACY_PRODUCT_NAME_RE = re.compile(r"OSM\s*GeoAgent|\bGeoAgent\b", re.I)


class _IdleAgent:
    async def run(self, request):  # type: ignore[no-untyped-def]
        return GeoAgentResponse(
            answer="unused",
            trace=[TraceEvent(kind="final_answer", message="unused")],
            stop_reason="final_answer",
            model="fake",
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent_test",
        ollama_model="deepseek-r1:7b",
        _env_file=None,  # type: ignore[call-arg]
    )


def _services(settings: Settings) -> ApplicationServices:
    registry = MagicMock()
    registry.names = ()
    return ApplicationServices(
        settings=settings,
        database=MagicMock(),
        embedding_provider=MagicMock(is_loaded=False),
        query_osm_tool=MagicMock(),
        resolve_place_tool=MagicMock(),
        analyze_features_tool=MagicMock(),
        llm_provider=MagicMock(),
        tool_registry=registry,
        geo_agent=_IdleAgent(),  # type: ignore[arg-type]
    )


@pytest.fixture
async def client(settings: Settings):
    app = create_app(settings, services=_services(settings))
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        yield http


def _read(*parts: str) -> str:
    return (_WEB_ROOT.joinpath(*parts)).read_text(encoding="utf-8")


async def test_ui_route_returns_html(client: httpx.AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Ariadne Thread" in response.text
    assert "Content-Security-Policy" in response.headers


async def test_browser_title_is_ariadne_thread(client: httpx.AsyncClient):
    html = (await client.get("/")).text
    match = re.search(r"<title>\s*([^<]+?)\s*</title>", html, flags=re.I)
    assert match is not None
    assert "Ariadne Thread" in match.group(1)


async def test_main_heading_is_ariadne_thread(client: httpx.AsyncClient):
    html = (await client.get("/")).text
    match = re.search(r"<h1>\s*([^<]+?)\s*</h1>", html, flags=re.I)
    assert match is not None
    assert "Ariadne Thread" in match.group(1)


async def test_logo_and_favicon_are_wired(client: httpx.AsyncClient):
    html = (await client.get("/")).text
    assert 'src="/static/img/ariadne-icon.png"' in html
    assert 'alt="Ariadne Thread logo"' in html
    assert 'href="/static/img/ariadne-icon.png"' in html
    assert 'class="brand-logo"' in html


async def test_legacy_product_name_not_visible_in_ui(client: httpx.AsyncClient):
    html = (await client.get("/")).text
    chrome = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.I | re.S)
    chrome = re.sub(r"<style\b[^>]*>.*?</style>", "", chrome, flags=re.I | re.S)
    assert _LEGACY_PRODUCT_NAME_RE.search(chrome) is None


async def test_comparison_report_section_is_present_but_hidden(client: httpx.AsyncClient):
    response = await client.get("/")
    html = response.text
    assert "Comparison Report" in html
    assert "Analysis Method" in html
    assert 'id="analysis-section"' in html
    assert "hidden" in html
    js = (await client.get("/static/js/main.js")).text
    assert "renderAnalysis" in js
    assert "renderEnergyAnalysis" in js
    assert "renderAnalysisCharts" in js
    assert "comparison-report" in js
    assert "Energy Analysis Results" in html
    assert "Analysis Charts" in html
    assert 'id="energy-analysis-section"' in html
    assert 'id="analysis-charts-section"' in html


async def test_analysis_charts_renderer_handles_bar_line_and_malformed():
    charts_js = _read("static", "js", "charts.js")
    html = _INDEX.read_text(encoding="utf-8")
    assert "export function renderAnalysisChart" in charts_js
    assert "export function renderAnalysisCharts" in charts_js
    assert 'id="analysis-charts-section"' in html
    assert "Analysis Charts" in html

    import subprocess
    import tempfile
    from pathlib import Path as _Path

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = _Path(tmp)
            (tmp_path / "charts.mjs").write_text(charts_js, encoding="utf-8")
            probe = tmp_path / "probe.mjs"
            probe.write_text(
                """
import { isRenderableChart, renderAnalysisCharts, renderAnalysisChart } from "./charts.mjs";

if (typeof document === "undefined") {
  const makeNode = (name) => {
    const node = {
      tagName: String(name).toUpperCase(),
      className: "",
      hidden: false,
      dataset: {},
      children: [],
      style: {},
      textContent: "",
      attrs: {},
      appendChild(child) {
        this.children.push(child);
        return child;
      },
      replaceChildren() {
        this.children = [];
      },
      setAttribute(key, value) {
        this.attrs[key] = String(value);
      },
      addEventListener() {},
    };
    return node;
  };
  globalThis.document = {
    createElement(name) {
      return makeNode(name);
    },
    createElementNS(_ns, name) {
      return makeNode(name);
    },
  };
}

const root = { replaceChildren() { this.cleared = true; }, children: [], cleared: false };
const none = renderAnalysisCharts([], root, { hidden: false });
if (none !== 0) throw new Error("empty charts should render nothing");

const bar = {
  chart_id: "pca_explained_variance",
  title: "PCA Explained Variance",
  chart_type: "bar",
  x_label: "Principal component",
  y_label: "Explained variance",
  series: [{ name: "Explained variance", data: [{ x: "PC1", y: 0.61 }, { x: "PC2", y: 0.27 }] }],
  metadata: { engine: "GeoLoadST", capability_id: "multidim_pca_clustering" },
};
const line = {
  chart_id: "trend",
  title: "Line",
  chart_type: "line",
  series: [{ name: "I", data: [{ x: 1, y: 0.2 }, { x: 2, y: 0.3 }] }],
};
if (!isRenderableChart(bar)) throw new Error("bar rejected");
if (!isRenderableChart(line)) throw new Error("line rejected");
const bad = { chart_type: "pie", title: "p", series: [] };
if (isRenderableChart(bad)) throw new Error("malformed accepted");
const card = renderAnalysisChart(bar);
if (!card.dataset || card.dataset.chartType !== "bar") throw new Error("bar card missing");
""",
                encoding="utf-8",
            )
            subprocess.run(["node", probe], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        assert "renderAnalysisChart" in charts_js
    except subprocess.CalledProcessError as exc:
        raise AssertionError(exc.stderr or exc.stdout) from exc


async def test_primary_english_labels_exist(client: httpx.AsyncClient):
    html = (await client.get("/")).text
    for label in [
        "Ariadne Thread",
        "Knowledge-grounded OpenStreetMap search and analysis",
        "Ask a geographic question",
        "Submit question",
        "Clear",
        "Answer",
        "Analysis Workflow",
        "OSM Documentation Sources",
        "Warnings",
        "Errors",
        "Generated Overpass Query",
        "Live OSM Results",
        "GeoJSON Preview",
        "Download GeoJSON",
        "Copy query",
        "Find public parks around Istanbul Technical University. Return at most 20 features.",
    ]:
        assert label in html


async def test_no_persian_ui_labels_in_page_chrome(client: httpx.AsyncClient):
    html = (await client.get("/")).text
    # Strip script/style blocks; remaining markup/copy must stay English.
    # Example query buttons may contain Persian user-input samples.
    stripped = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.I | re.S)
    stripped = re.sub(r"<style\b[^>]*>.*?</style>", "", stripped, flags=re.I | re.S)
    stripped = re.sub(
        r'<div id="examples"[\s\S]*?</div>',
        "",
        stripped,
        count=1,
    )
    assert _PERSIAN_UI_RE.search(stripped) is None
    assert re.search(r"[A-Za-z]", stripped) is not None


async def test_form_targets_canonical_api_contract(client: httpx.AsyncClient):
    html = (await client.get("/")).text
    assert 'data-api-path="/api/v1/agent/query"' in html
    api_js = _read("static", "js", "api.js")
    main_js = _read("static", "js", "main.js")
    assert 'AGENT_QUERY_PATH = "/api/v1/agent/query"' in api_js
    assert 'method: "POST"' in api_js
    assert "conversationId" in api_js or "conversation_id" in api_js
    assert "ariadne-conversation-id" in main_js
    assert "selected_indicator_id" in main_js
    assert "osm_grounding" in main_js


async def test_workflow_shows_execution_memory_decisions():
    api_js = _read("static", "js", "api.js")
    main_js = _read("static", "js", "main.js")
    assert '"memory_reuse"' in api_js
    # Memory steps render their own message so reuse is readable in the trace.
    assert 'event === "memory_reuse" && message' in main_js
    assert "memory_step" in main_js
    assert '"memory step"' in main_js
    assert "workflowMetaRows" in main_js


async def test_istanbul_examples_and_initial_map_viewport():
    main_js = _read("static", "js", "main.js")
    map_js = _read("static", "js", "map.js")
    css = _CSS.read_text(encoding="utf-8")
    assert "larger proportion of green space" in main_js
    assert "Boğaziçi University" in main_js
    assert "Yıldız Technical University" in main_js  # noqa: RUF001
    assert "DEFAULT_CENTER = [28.9784, 41.0082]" in map_js
    assert "DEFAULT_ZOOM = 11" in map_js
    assert "example-chip" in css
    assert "overflow-wrap: anywhere" in css


async def test_model_chip_uses_health_provider_and_model():
    main_js = _read("static", "js", "main.js")
    assert "health.llm_model" in main_js
    assert "health.llm_provider" in main_js
    assert "Model:" in main_js


async def test_static_assets_are_served(client: httpx.AsyncClient):
    css = await client.get("/static/css/app.css")
    assert css.status_code == 200
    assert "font-family" in css.text
    assert "--bg:" in css.text
    # Light SaaS surface — not the previous heavy dark body.
    assert "#f4f7fb" in css.text or "f4f7fb" in css.text
    assert "grid-template-columns: 1fr 1fr" in css.text
    assert "@media (min-width: 640px)" in css.text
    assert "@media (max-width: 639px)" in css.text

    for name in ("main.js", "api.js", "dom.js", "map.js", "download.js"):
        response = await client.get(f"/static/js/{name}")
        assert response.status_code == 200, name
        assert response.text.strip()

    plugin = await client.get("/static/vendor/mapbox-gl-rtl-text-0.2.3.js")
    assert plugin.status_code == 200
    assert "registerRTLTextPlugin" in plugin.text


async def test_logo_and_favicon_are_served(client: httpx.AsyncClient):
    html = (await client.get("/")).text
    assert 'src="/static/img/ariadne-icon.png"' in html
    assert 'href="/static/img/ariadne-icon.png"' in html
    assert 'alt="Ariadne Thread logo"' in html

    logo = await client.get("/static/img/ariadne-icon.png")
    assert logo.status_code == 200
    assert logo.headers["content-type"].startswith("image/")
    assert logo.content[:8] == b"\x89PNG\r\n\x1a\n"


async def test_documentation_only_branch_clears_map_features():
    main_js = _read("static", "js", "main.js")
    assert "No live map data was requested for this question." in main_js
    assert "clearGeoJson()" in main_js
    assert "download-geojson-btn" in main_js
    assert "liveQueryExecuted" in main_js


async def test_geojson_is_passed_to_map_layer():
    main_js = _read("static", "js", "main.js")
    map_js = _read("static", "js", "map.js")
    assert "setGeoJson(collection)" in main_js
    assert "source.setData(painted)" in map_js
    assert 'RESULTS_SOURCE_ID = "ariadne-results"' in map_js
    assert 'ANALYSIS_SOURCE_ID = "ariadne-analysis-area"' in map_js
    assert 'LISA_SOURCE_ID = "ariadne-lisa-clusters"' in map_js
    assert 'ENERGY_SOURCE_ID = "ariadne-energy-overlay"' in map_js
    assert "ariadne-lisa-point" in map_js
    assert "ariadne-energy-point" in map_js
    assert "splitLisaCollection" in map_js
    assert "splitEnergyLayers" in map_js
    assert "HIGH_HIGH" in map_js


async def test_map_layers_cover_point_line_polygon():
    map_js = _read("static", "js", "map.js")
    html = _INDEX.read_text(encoding="utf-8")
    assert "ariadne-point" in map_js
    assert "ariadne-line" in map_js
    assert "ariadne-fill" in map_js
    assert "ariadne-fill-outline" in map_js
    assert '["==", "$type", "Point"]' in map_js
    assert '["==", "$type", "LineString"]' in map_js
    assert '["==", "$type", "Polygon"]' in map_js
    assert '"fill-outline-color"' not in map_js
    assert '["get", "target_id"]' in map_js
    assert "mapLibreProperties" in map_js
    assert "fitBounds" in map_js
    assert "Preserve current view for empty results" in map_js or "empty results" in map_js
    assert "TARGET_PALETTE" in map_js
    assert "target_id" in map_js
    assert "analysis_target_label" in map_js
    assert "map-target-legend" in html
    assert "map-lisa-legend" in html
    assert "map-energy-legend" in html
    assert "High-High hotspot" in html
    assert "Low-Low coldspot" in html
    assert "High-Low outlier" in html
    assert "Low-High outlier" in html
    assert "targetsFromCollection" in map_js
    assert "for (const feature of collection.features)" in map_js
    assert "walkCoords" in map_js
    assert "updateCompositionLegend(painted, energyPainted)" in map_js
    assert "flattenFeatureProperties" in map_js
    assert "id: index + 1" in map_js
    assert "generateId: false" in map_js


async def test_comparison_map_keeps_all_target_features():
    map_js = _read("static", "js", "map.js")
    html = _INDEX.read_text(encoding="utf-8")
    css = _CSS.read_text(encoding="utf-8")
    main_js = _read("static", "js", "main.js")
    assert "mapLibreCollection" in map_js
    assert "assignTargetId" in map_js
    assert 'id="sources-section"' in html
    assert "workflow-list" in html
    assert "workflow-card" in main_js or "workflow-list" in main_js
    assert "workflowMetaRows" in main_js
    assert '"memory step"' in main_js
    assert "workflow-meta-row" in main_js
    assert "title.title = titleText" in main_js
    assert "source-url" in main_js
    energy_at = html.find('id="energy-analysis-heading"')
    charts_at = html.find('id="analysis-charts-heading"')
    answer_at = html.find('id="answer-heading"')
    workflow_at = html.find('id="workflow-heading"')
    sources_at = html.find('id="sources-heading"')
    assert 0 < energy_at < charts_at < answer_at < workflow_at < sources_at
    assert "Energy Analysis Results" in html
    assert "Analysis Charts" in html
    assert "renderEnergyAnalysis" in main_js
    assert "renderAnalysisCharts" in main_js
    section_at = html.find('id="analysis-charts-section"')
    assert "hidden" in html[max(0, section_at - 80) : section_at]
    assert "workflow-sources-grid" not in html
    assert ".workflow-list," in css or ".workflow-list" in css
    assert "line-clamp: 2" in css
    assert "overflow-wrap: anywhere" in css
    assert "word-break: break-word" in css

    import subprocess
    import tempfile
    from pathlib import Path as _Path

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = _Path(tmp)
            (tmp_path / "map.mjs").write_text(map_js, encoding="utf-8")
            probe = tmp_path / "probe.mjs"
            probe.write_text(
                """
import {
  mapLibreCollection,
  mapLibreLisaCollection,
  mapLibreEnergyCollection,
  splitLisaCollection,
  splitEnergyLayers,
  targetsFromCollection,
  TARGET_PALETTE,
  LISA_CLUSTER_COLORS,
} from "./map.mjs";

const collection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      id: 99,
      geometry: { type: "Point", coordinates: [51.395, 35.702] },
      properties: {
        osm_id: 99,
        ariadne: { target_id: "t1" },
        analysis_target_label: "University of Tehran",
      },
    },
    {
      type: "Feature",
      id: 99,
      geometry: { type: "Point", coordinates: [51.351, 35.703] },
      properties: {
        osm_id: 99,
        target_id: "t2",
        analysis_target_label: "Sharif University of Technology",
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [51.4, 35.7] },
      properties: { osm_id: 1, analysis_target_label: "University of Tehran" },
    },
  ],
};
const painted = mapLibreCollection(collection);
if (painted.features.length !== 3) throw new Error("dropped features");
const ids = new Set(painted.features.map((f) => f.id));
if (ids.size !== 3) throw new Error("feature ids collided");
const targetIds = painted.features.map((f) => f.properties.target_id);
if (targetIds[0] !== "t1" || targetIds[1] !== "t2") throw new Error("target ids lost");
if (targetIds[2] !== "t1") throw new Error("label was not mapped to target_id");
const legend = targetsFromCollection(painted);
if (legend.length !== 2) throw new Error("legend missing a target");
if (legend[0].color === legend[1].color) throw new Error("targets share a colour");
if (!TARGET_PALETTE.includes(legend[0].color)) throw new Error("unexpected colour");

const mixed = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      geometry: { type: "LineString", coordinates: [[10, 53], [11, 54]] },
      properties: { name: "line", target_id: "t1" },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [10.1, 53.2] },
      properties: {
        cluster_type: "HIGH_HIGH",
        indicator: "LISA",
        value: 0.82,
        p_value: 0.01,
        layer_name: "GeoLoadST LISA Clusters",
      },
    },
  ],
};
const split = splitLisaCollection(mixed);
if (split.topology.features.length !== 1) throw new Error("topology lost");
if (split.lisa.features.length !== 1) throw new Error("lisa cluster lost");
if (split.topology.features[0].geometry.type !== "LineString") {
  throw new Error("lisa was mixed into topology");
}
const lisaPainted = mapLibreLisaCollection(split.lisa);
if (lisaPainted.features[0].properties.cluster_type !== "HIGH_HIGH") {
  throw new Error("cluster_type stripped");
}
if (lisaPainted.features[0].properties.indicator !== "LISA") {
  throw new Error("indicator stripped");
}
if (LISA_CLUSTER_COLORS.HIGH_HIGH !== "#dc2626") throw new Error("hh colour");

const energyMixed = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      geometry: { type: "LineString", coordinates: [[10, 53], [11, 54]] },
      properties: { name: "line", target_id: "t1", target_label: "SimBench Network" },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [10.1, 53.2] },
      properties: {
        bus_id: 7,
        degree_centrality: 0.2,
        betweenness_centrality: 0.9,
        closeness_centrality: 0.4,
        analysis: "topology_centrality",
        layer_name: "GeoLoadST Topology Centrality",
        in_top_n: true,
        top_n: 10,
        rank: 1,
      },
    },
  ],
};
const energySplit = splitEnergyLayers(energyMixed);
if (energySplit.topology.features.length !== 1) throw new Error("simbench topology lost");
if (energySplit.energy.features.length !== 1) throw new Error("energy overlay lost");
if (energySplit.topology.features[0].geometry.type !== "LineString") {
  throw new Error("overlay was mixed into the network layer");
}
const energyPainted = mapLibreEnergyCollection(energySplit.energy);
if (energyPainted.features[0].properties.degree_centrality !== 0.2) {
  throw new Error("degree stripped");
}
if (energyPainted.features[0].properties.betweenness_centrality !== 0.9) {
  throw new Error("betweenness stripped");
}
if (energyPainted.features[0].properties.energy_overlay !== "yes") {
  throw new Error("overlay flag missing");
}
if (energyPainted.features[0].properties.layer_name !== "GeoLoadST Topology Centrality") {
  throw new Error("layer name lost");
}
if (energyPainted.features[0].properties.viz_norm !== 0.5) {
  throw new Error("single-value overlay must keep a mid visualization scale");
}
""",
                encoding="utf-8",
            )
            subprocess.run(
                ["node", probe],
                check=True,
                capture_output=True,
                text=True,
            )
    except FileNotFoundError:
        assert "mapLibreCollection" in map_js
    except subprocess.CalledProcessError as exc:
        raise AssertionError(exc.stderr or exc.stdout) from exc


async def test_maplibre_rtl_plugin_is_registered_once():
    map_js = _read("static", "js", "map.js")
    html = _INDEX.read_text(encoding="utf-8")
    assert "maplibre-gl@4.7.1" in html
    assert "setRTLTextPlugin" in map_js
    assert "getRTLTextPluginStatus" in map_js
    assert 'RTL_TEXT_PLUGIN_VERSION = "0.2.3"' in map_js
    assert "/static/vendor/mapbox-gl-rtl-text-${RTL_TEXT_PLUGIN_VERSION}.js" in map_js
    assert "ensureRtlTextPlugin" in map_js
    assert "if (rtlPluginPromise) return rtlPluginPromise" in map_js
    assert "createMap(containerId)" in map_js
    assert "await ensureRtlTextPlugin()" in map_js
    assert "unpkg.com/@mapbox/mapbox-gl-rtl-text@latest" not in map_js
    plugin = _WEB_ROOT / "static" / "vendor" / "mapbox-gl-rtl-text-0.2.3.js"
    assert plugin.is_file()
    plugin_text = plugin.read_text(encoding="utf-8")
    assert "registerRTLTextPlugin" in plugin_text
    assert "applyArabicShaping" in plugin_text
    assert "processBidirectionalText" in plugin_text
    notice = (_WEB_ROOT / "static" / "vendor" / "NOTICE").read_text(encoding="utf-8")
    assert "0.2.3" in notice
    assert "BSD-2-Clause" in notice


async def test_no_persian_string_reversal_or_map_rtl_css_hacks():
    map_js = _read("static", "js", "map.js")
    css = _CSS.read_text(encoding="utf-8")
    html = _INDEX.read_text(encoding="utf-8")
    for name in ("main.js", "api.js", "dom.js", "map.js", "download.js"):
        text = _read("static", "js", name)
        assert 'split("").reverse()' not in text
        assert "split('').reverse()" not in text
        assert "name:fa" not in text or name == "map.js"
    assert ".split(" not in map_js or 'split("").reverse()' not in map_js
    assert "direction: rtl" not in css
    assert "direction:rtl" not in css.replace(" ", "")
    assert 'dir="rtl"' not in html
    assert 'lang="en"' in html
    assert 'dir="auto"' in map_js
    canvas_block = re.search(r"\.map-canvas\s*\{[^}]+\}", css)
    assert canvas_block is not None
    assert "direction" not in canvas_block.group(0)


async def test_rtl_plugin_failure_does_not_block_map_init():
    map_js = _read("static", "js", "map.js")
    assert "RTL text plugin failed to load" in map_js
    assert "console.warn" in map_js
    assert "if (!map) createMap(containerId)" in map_js
    assert "TARGET_PALETTE" in map_js
    assert 'RESULTS_SOURCE_ID = "ariadne-results"' in map_js
    assert "mapLibreProperties" in map_js
    assert '["get", "target_id"]' in map_js


async def test_empty_feature_collection_handled():
    main_js = _read("static", "js", "main.js")
    map_js = _read("static", "js", "map.js")
    assert "no matching features were found" in main_js
    assert "emptyFeatureCollection" in map_js
    assert 'type: "FeatureCollection"' in map_js


async def test_geojson_download_contract():
    download_js = _read("static", "js", "download.js")
    main_js = _read("static", "js", "main.js")
    assert 'GEOJSON_MIME = "application/geo+json"' in download_js
    assert "JSON.stringify(geojson, null, 2)" in download_js
    assert ".geojson" in download_js
    assert "URL.revokeObjectURL" in download_js
    assert "downloadGeoJson" in main_js
    assert "Preparing download..." in main_js
    assert "Downloaded" in main_js
    assert "Download failed" in main_js
    assert 'els["download-geojson-btn"].disabled = true' in main_js


async def test_analysis_workflow_and_local_map_event():
    main_js = _read("static", "js", "main.js")
    html = _INDEX.read_text(encoding="utf-8")
    css = _CSS.read_text(encoding="utf-8")
    assert "Analysis Workflow" in html
    assert "Rendered GeoJSON on the map" in main_js
    assert "Client-side event" in main_js
    assert "PROTOCOL_LEAK_RE" in main_js
    assert "workflowMetaRows" in main_js
    assert "workflow-meta-row" in main_js
    assert '"memory step"' in main_js
    assert "title.title = titleText" in main_js
    assert "grid-template-columns: 1fr 1fr" in css
    assert ".workflow-list" in css
    assert ".sources" in css
    assert "line-clamp: 2" in css
    assert "overflow-wrap: anywhere" in css
    assert "word-break: break-word" in css
    workflow_at = html.find('id="workflow-heading"')
    sources_at = html.find('id="sources-heading"')
    assert workflow_at < sources_at
    assert "workflow-sources-grid" not in html


async def test_answer_panel_blocks_protocol_leakage():
    main_js = _read("static", "js", "main.js")
    assert "sanitizeAnswer" in main_js
    assert "tool_calls" in main_js
    assert "invalid structured response" in main_js
    assert "innerHTML" not in main_js


async def test_answer_placeholder_is_hidden_when_answer_exists():
    css = _CSS.read_text(encoding="utf-8")
    main_js = _read("static", "js", "main.js")
    assert "[hidden]" in css
    assert "display: none !important" in css
    assert 'els["answer-empty"].hidden = true' in main_js
    assert "Request failed" in main_js
    assert "live_query_failed" in main_js
    assert "liveErrorLabel" in main_js
    assert "protocol_repair" in _read("static", "js", "api.js")


async def test_api_error_messages_are_english_and_safe():
    api_js = _read("static", "js", "api.js")
    for needle in [
        "rate-limited",
        "temporarily unavailable",
        "timed out",
        "failed validation",
        "malformed JSON",
        "Network failure",
    ]:
        assert needle in api_js
    assert "traceback" not in api_js.lower()
    assert "DATABASE_URL" not in api_js
    assert "password" not in api_js.lower()


async def test_model_output_uses_text_not_trusted_html():
    main_js = _read("static", "js", "main.js")
    dom_js = _read("static", "js", "dom.js")
    assert "appendLines(els.answer, answer)" in main_js
    assert "innerHTML" not in main_js
    assert "el.textContent" in dom_js
    assert "createTextNode" in dom_js


async def test_source_urls_are_validated():
    dom_js = _read("static", "js", "dom.js")
    main_js = _read("static", "js", "main.js")
    assert "export function isSafeHttpUrl" in dom_js
    assert 'url.protocol !== "http:"' in dom_js
    assert "url.username || url.password" in dom_js
    assert "isSafeHttpUrl(source.url)" in main_js
    assert 'rel = "noopener noreferrer"' in dom_js


async def test_execution_trace_filters_hidden_reasoning():
    main_js = _read("static", "js", "main.js")
    assert "<think" in main_js  # filter condition
    assert "continue;" in main_js
    assert "labelForTraceEvent" in main_js


async def test_overpass_query_is_readonly_without_execution_control(client: httpx.AsyncClient):
    html = (await client.get("/")).text
    assert "Generated Overpass Query" in html
    assert "cannot execute Overpass QL" in html
    assert "Execute Overpass" not in html
    assert "run overpass" not in html.lower()
    main_js = _read("static", "js", "main.js")
    assert "Copy query" in main_js
    assert "executeOverpass" not in main_js


async def test_osm_attribution_present(client: httpx.AsyncClient):
    html = (await client.get("/")).text
    assert "OpenStreetMap contributors" in html
    assert "openstreetmap.org/copyright" in html
    map_js = _read("static", "js", "map.js")
    assert "attributionControl: true" in map_js


async def test_ui_route_does_not_load_embeddings(client: httpx.AsyncClient, settings: Settings):
    app = create_app(settings, services=_services(settings))
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        await http.get("/")
        assert app.state.embedding_provider.is_loaded is False


async def test_javascript_syntax_parses():
    import subprocess
    import tempfile
    from pathlib import Path as _Path

    for name in ("main.js", "api.js", "dom.js", "map.js", "download.js"):
        path = _JS_DIR / name
        text = path.read_text(encoding="utf-8")
        assert "import " in text or "export " in text
        try:
            with tempfile.TemporaryDirectory() as tmp:
                module_path = _Path(tmp) / f"{name}.mjs"
                module_path.write_text(text, encoding="utf-8")
                subprocess.run(
                    ["node", "--check", str(module_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
        except FileNotFoundError:
            # No Node runtime in the offline suite environment.
            assert "function " in text or "=>" in text
        except subprocess.CalledProcessError as exc:
            raise AssertionError(f"{name} failed node --check: {exc.stderr}") from exc


async def test_index_and_assets_exist_on_disk():
    assert _INDEX.is_file()
    assert _CSS.is_file()
    for name in ("main.js", "api.js", "dom.js", "map.js", "download.js"):
        assert (_JS_DIR / name).is_file()
    vendor = _WEB_ROOT / "static" / "vendor"
    assert (vendor / "mapbox-gl-rtl-text-0.2.3.js").is_file()
    img_dir = _WEB_ROOT / "static" / "img"
    for name in ("ariadne-icon.png", "favicon.png", "apple-touch-icon.png"):
        assert (img_dir / name).is_file()
