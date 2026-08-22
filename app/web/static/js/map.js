/**
 * MapLibre rendering for live OSM GeoJSON and trusted analysis scopes.
 */

export const RESULTS_SOURCE_ID = "ariadne-results";
export const ANALYSIS_SOURCE_ID = "ariadne-analysis-area";

const LAYER_FILL = "ariadne-fill";
const LAYER_OUTLINE = "ariadne-fill-outline";
const LAYER_LINE = "ariadne-line";
const LAYER_POINT = "ariadne-point";
const LAYER_ANALYSIS = "ariadne-analysis-outline";

/** Distinct colours for comparison target groups (t1, t2, …). */
export const TARGET_PALETTE = ["#0f766e", "#c026d3", "#b45309", "#1d4ed8"];
const SHARED_COLOR = "#ca8a04";
const FALLBACK_COLOR = "#38bdf8";

/** Token-free MapLibre style (OpenFreeMap Liberty). */
export const BASEMAP_STYLE = "https://tiles.openfreemap.org/styles/liberty";

/**
 * MapLibre GL JS 4.7.1 RTL plugin (@mapbox/mapbox-gl-rtl-text 0.2.3).
 * Served from Ariadne static assets (not an unpinned CDN). Required for
 * Arabic-script shaping and bidi of Persian/Arabic basemap labels.
 */
export const RTL_TEXT_PLUGIN_VERSION = "0.2.3";
export const RTL_TEXT_PLUGIN_PATH = `/static/vendor/mapbox-gl-rtl-text-${RTL_TEXT_PLUGIN_VERSION}.js`;

const RTL_PLUGIN_LOAD_TIMEOUT_MS = 8000;

/** Istanbul metropolitan area (lng, lat). Not a campus. */
const DEFAULT_CENTER = [28.9784, 41.0082];
const DEFAULT_ZOOM = 11;

let map = null;
let popup = null;
let styleReady = false;
let pendingResults = null;
let pendingAnalysis = null;
let lastRenderedCollection = null;
let mapInitStarted = false;
let rtlPluginPromise = null;

export function initMap(containerId) {
  if (map) return map;
  if (typeof maplibregl === "undefined") {
    throw new Error("MapLibre GL JS failed to load");
  }
  if (!mapInitStarted) {
    mapInitStarted = true;
    void startMap(containerId);
  }
  return map;
}

async function startMap(containerId) {
  await ensureRtlTextPlugin();
  if (!map) createMap(containerId);
}

/**
 * Register MapLibre's RTL text plugin once. Failures warn in the console and
 * still allow the map to load (English labels remain usable).
 */
export function ensureRtlTextPlugin() {
  if (rtlPluginPromise) return rtlPluginPromise;
  rtlPluginPromise = registerRtlTextPluginOnce();
  return rtlPluginPromise;
}

async function registerRtlTextPluginOnce() {
  if (typeof maplibregl.setRTLTextPlugin !== "function") {
    console.warn(
      "ariadne-map: MapLibre RTL text plugin API is unavailable; Arabic-script labels may render incorrectly",
    );
    return "unavailable";
  }
  const status = rtlPluginStatus();
  if (status === "loaded") return "loaded";
  if (status === "error") {
    console.warn(
      "ariadne-map: RTL text plugin previously failed; Arabic-script labels may render incorrectly",
    );
    return "error";
  }
  if (status === "loading") {
    return waitForRtlPlugin(RTL_PLUGIN_LOAD_TIMEOUT_MS);
  }
  try {
    await Promise.race([
      maplibregl.setRTLTextPlugin(absoluteUrl(RTL_TEXT_PLUGIN_PATH), false),
      timeoutMs(RTL_PLUGIN_LOAD_TIMEOUT_MS),
    ]);
    const next = rtlPluginStatus();
    if (next === "loaded") return "loaded";
    if (next === "error") {
      console.warn(
        "ariadne-map: RTL text plugin failed to load; Arabic-script labels may render incorrectly",
      );
      return "error";
    }
    if (next === "loading") {
      console.warn(
        "ariadne-map: RTL text plugin load timed out; Arabic-script labels may render incorrectly",
      );
      return "timeout";
    }
    return next || "loaded";
  } catch (error) {
    console.warn(
      "ariadne-map: RTL text plugin failed to load; Arabic-script labels may render incorrectly",
      error,
    );
    return "error";
  }
}

function rtlPluginStatus() {
  if (typeof maplibregl.getRTLTextPluginStatus !== "function") return "unavailable";
  return maplibregl.getRTLTextPluginStatus();
}

function waitForRtlPlugin(ms) {
  return new Promise((resolve) => {
    const started = Date.now();
    const tick = () => {
      const status = rtlPluginStatus();
      if (status === "loaded") {
        resolve("loaded");
        return;
      }
      if (status === "error") {
        console.warn(
          "ariadne-map: RTL text plugin previously failed; Arabic-script labels may render incorrectly",
        );
        resolve("error");
        return;
      }
      if (Date.now() - started >= ms) {
        console.warn(
          "ariadne-map: RTL text plugin load timed out; Arabic-script labels may render incorrectly",
        );
        resolve("timeout");
        return;
      }
      setTimeout(tick, 50);
    };
    tick();
  });
}

function timeoutMs(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function absoluteUrl(path) {
  if (typeof window === "undefined" || !window.location) return path;
  return new URL(path, window.location.origin).href;
}

function createMap(containerId) {
  map = new maplibregl.Map({
    container: containerId,
    style: BASEMAP_STYLE,
    center: DEFAULT_CENTER,
    zoom: DEFAULT_ZOOM,
    attributionControl: true,
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  popup = new maplibregl.Popup({ closeButton: true, maxWidth: "280px" });
  const onStyleReady = () => {
    styleReady = true;
    ensureLayers();
    if (pendingResults) {
      applyResults(pendingResults);
      pendingResults = null;
    } else if (lastRenderedCollection) {
      applyResults(lastRenderedCollection);
    }
    if (pendingAnalysis) {
      applyAnalysis(pendingAnalysis);
      pendingAnalysis = null;
    }
  };
  map.on("load", onStyleReady);
  map.on("style.load", onStyleReady);
  return map;
}

function ensureLayers() {
  if (!map) return;
  if (!map.getSource(RESULTS_SOURCE_ID)) {
    map.addSource(RESULTS_SOURCE_ID, {
      type: "geojson",
      data: emptyFeatureCollection(),
      generateId: false,
    });
  }
  if (!map.getSource(ANALYSIS_SOURCE_ID)) {
    map.addSource(ANALYSIS_SOURCE_ID, {
      type: "geojson",
      data: emptyFeatureCollection(),
    });
  }

  // Use legacy $type filters (Polygon also matches MultiPolygon). Do not mix
  // $type with ["geometry-type"] in one filter — that compiles as an
  // expression where "$type" is a string and can exclude every feature.
  addLayerSafe({
    id: LAYER_FILL,
    type: "fill",
    source: RESULTS_SOURCE_ID,
    filter: ["==", "$type", "Polygon"],
    paint: {
      "fill-color": targetColorExpression(),
      "fill-opacity": 0.42,
    },
  });
  addLayerSafe({
    id: LAYER_OUTLINE,
    type: "line",
    source: RESULTS_SOURCE_ID,
    filter: ["==", "$type", "Polygon"],
    paint: {
      "line-color": targetColorExpression(),
      "line-width": 1.8,
      "line-opacity": 1,
    },
  });
  addLayerSafe({
    id: LAYER_LINE,
    type: "line",
    source: RESULTS_SOURCE_ID,
    filter: ["==", "$type", "LineString"],
    paint: {
      "line-color": targetColorExpression(),
      "line-width": 2.6,
      "line-opacity": 1,
    },
  });
  addLayerSafe({
    id: LAYER_POINT,
    type: "circle",
    source: RESULTS_SOURCE_ID,
    filter: ["==", "$type", "Point"],
    paint: {
      "circle-radius": 6,
      "circle-color": targetColorExpression(),
      "circle-opacity": 1,
      "circle-stroke-width": 1.4,
      "circle-stroke-color": "#0b1220",
    },
  });
  addLayerSafe({
    id: LAYER_ANALYSIS,
    type: "line",
    source: ANALYSIS_SOURCE_ID,
    paint: {
      "line-color": "#818cf8",
      "line-width": 2,
      "line-dasharray": [2, 2],
    },
  });

  for (const layerId of [LAYER_POINT, LAYER_LINE, LAYER_OUTLINE, LAYER_FILL]) {
    map.off("click", layerId, onFeatureClick);
    map.on("click", layerId, onFeatureClick);
    map.off("mouseenter", layerId, onEnter);
    map.on("mouseenter", layerId, onEnter);
    map.off("mouseleave", layerId, onLeave);
    map.on("mouseleave", layerId, onLeave);
  }
}

function addLayerSafe(layer) {
  if (!map || map.getLayer(layer.id)) return;
  const beforeId = firstSymbolLayerId();
  try {
    if (beforeId) map.addLayer(layer, beforeId);
    else map.addLayer(layer);
  } catch (error) {
    console.warn("ariadne-map: failed to add layer", layer.id, error);
  }
}

function firstSymbolLayerId() {
  if (!map) return undefined;
  const style = map.getStyle();
  const layers = (style && style.layers) || [];
  const found = layers.find((layer) => layer.type === "symbol");
  return found ? found.id : undefined;
}

function onFeatureClick(event) {
  const feature = event.features && event.features[0];
  if (!feature || !popup || !map) return;
  popup.setLngLat(event.lngLat).setHTML(popupHtml(feature)).addTo(map);
}

function onEnter() {
  if (map) map.getCanvas().style.cursor = "pointer";
}

function onLeave() {
  if (map) map.getCanvas().style.cursor = "";
}

export function emptyFeatureCollection() {
  return { type: "FeatureCollection", features: [] };
}

export function setGeoJson(geojson) {
  const data = normalizeFeatureCollection(geojson);
  lastRenderedCollection = data;
  if (!map) return { rendered: false, collection: data };
  if (!styleReady) {
    pendingResults = data;
    return { rendered: false, collection: data };
  }
  applyResults(data);
  return { rendered: true, collection: data };
}

export function clearGeoJson() {
  setGeoJson(emptyFeatureCollection());
  clearAnalysisArea();
}

export function getLastRenderedCollection() {
  return lastRenderedCollection;
}

function applyResults(data) {
  ensureLayers();
  const source = map.getSource(RESULTS_SOURCE_ID);
  // Paint the full FeatureCollection in one setData. Never replace with a
  // single-target subset — comparison copies (same osm_id, different target)
  // must all remain.
  const painted = mapLibreCollection(data);
  if (source) source.setData(painted);
  logRenderDiagnostics(data, painted);
  updateTargetLegend(painted);
  fitToFeatures(painted);
}

function logRenderDiagnostics(original, painted) {
  const geometry = { Point: 0, LineString: 0, Polygon: 0, other: 0 };
  const targetIds = new Set();
  for (const feature of painted.features) {
    const rawType = feature.geometry && feature.geometry.type;
    const key = String(rawType || "").replace(/^Multi/, "");
    if (key in geometry) geometry[key] += 1;
    else geometry.other += 1;
    const id = feature.properties && feature.properties.target_id;
    if (id) targetIds.add(String(id));
  }
  console.debug("ariadne-map", {
    features: original.features.length,
    painted: painted.features.length,
    geometry,
    targetIds: [...targetIds],
    source: Boolean(map && map.getSource(RESULTS_SOURCE_ID)),
    layers: [LAYER_FILL, LAYER_OUTLINE, LAYER_LINE, LAYER_POINT].filter(
      (id) => map && map.getLayer(id),
    ),
  });
}

function targetColorExpression() {
  // Style by top-level target_id (t1/t2). Polygon outlines use a line layer
  // because MapLibre rejects feature-data expressions on fill outline paint,
  // and a rejected fill layer would prevent later layers from being added.
  return [
    "match",
    ["downcase", ["to-string", ["coalesce", ["get", "target_id"], ""]]],
    "t1",
    TARGET_PALETTE[0],
    "t2",
    TARGET_PALETTE[1],
    "t3",
    TARGET_PALETTE[2],
    "t4",
    TARGET_PALETTE[3],
    "shared",
    SHARED_COLOR,
    FALLBACK_COLOR,
  ];
}

export function targetsFromCollection(collection) {
  const byId = new Map();
  const features = collection && Array.isArray(collection.features) ? collection.features : [];
  for (const feature of features) {
    const props = flattenFeatureProperties((feature && feature.properties) || {});
    const targetId = targetIdFromProps(props);
    const label =
      props.analysis_target_label || props.analysis_target || props.target_label || targetId;
    if (!targetId || !label) continue;
    if (!byId.has(targetId)) {
      byId.set(targetId, {
        id: targetId,
        label: String(label),
        color: colorForTargetId(targetId),
      });
    }
  }
  return [...byId.values()].sort((a, b) => a.id.localeCompare(b.id));
}

function targetIdFromProps(props) {
  const p = flattenFeatureProperties(props);
  if (p.target_id != null && String(p.target_id).trim()) {
    return String(p.target_id).trim();
  }
  if (typeof p.target_index === "number" && Number.isFinite(p.target_index) && p.target_index >= 0) {
    return `t${p.target_index + 1}`;
  }
  return "";
}

function colorForTargetId(targetId) {
  if (targetId === "shared") return SHARED_COLOR;
  const match = /^t(\d+)$/i.exec(targetId);
  if (!match) return FALLBACK_COLOR;
  const index = Number(match[1]) - 1;
  if (index < 0) return FALLBACK_COLOR;
  return TARGET_PALETTE[index % TARGET_PALETTE.length];
}

function updateTargetLegend(collection) {
  const list = document.getElementById("map-target-legend");
  if (!list) return;
  const targets = targetsFromCollection(collection);
  list.replaceChildren();
  if (targets.length < 2) {
    list.hidden = true;
    return;
  }
  list.hidden = false;
  for (const target of targets) {
    const li = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch legend-target";
    swatch.style.background = target.color;
    swatch.style.borderColor = target.color;
    li.appendChild(swatch);
    li.appendChild(document.createTextNode(target.label));
    list.appendChild(li);
  }
}

/**
 * Trusted analysis area only (point-radius / bbox). Never invents geometry.
 * @param {{ type: 'point', lat: number, lon: number, radius_m: number } | { type: 'bbox', south: number, west: number, north: number, east: number } | null} scope
 */
export function setAnalysisArea(scope) {
  if (!scope) {
    clearAnalysisArea();
    return;
  }
  const collection = analysisCollection(scope);
  if (!map) return;
  if (!styleReady) {
    pendingAnalysis = collection;
    return;
  }
  applyAnalysis(collection);
}

export function clearAnalysisArea() {
  pendingAnalysis = null;
  if (!map || !styleReady) return;
  ensureLayers();
  const source = map.getSource(ANALYSIS_SOURCE_ID);
  if (source) source.setData(emptyFeatureCollection());
}

function applyAnalysis(collection) {
  ensureLayers();
  const source = map.getSource(ANALYSIS_SOURCE_ID);
  if (source) source.setData(collection);
}

function analysisCollection(scope) {
  if (scope.type === "point") {
    return {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: { kind: "analysis_area", label: "Analysis Area" },
          geometry: {
            type: "Polygon",
            coordinates: [circleRing(scope.lon, scope.lat, scope.radius_m)],
          },
        },
      ],
    };
  }
  if (scope.type === "bbox") {
    const { west, south, east, north } = scope;
    return {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: { kind: "analysis_area", label: "Analysis Area" },
          geometry: {
            type: "Polygon",
            coordinates: [
              [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
              ],
            ],
          },
        },
      ],
    };
  }
  return emptyFeatureCollection();
}

function circleRing(lon, lat, radiusMeters, steps = 64) {
  const coords = [];
  const earth = 6378137;
  const latRad = (lat * Math.PI) / 180;
  const meterPerDegLat = (Math.PI / 180) * earth;
  const meterPerDegLon = meterPerDegLat * Math.cos(latRad);
  for (let i = 0; i <= steps; i += 1) {
    const angle = (i / steps) * Math.PI * 2;
    const dx = (radiusMeters * Math.cos(angle)) / meterPerDegLon;
    const dy = (radiusMeters * Math.sin(angle)) / meterPerDegLat;
    coords.push([lon + dx, lat + dy]);
  }
  return coords;
}

function normalizeFeatureCollection(geojson) {
  if (!geojson || typeof geojson !== "object") return emptyFeatureCollection();
  if (geojson.type !== "FeatureCollection" || !Array.isArray(geojson.features)) {
    return emptyFeatureCollection();
  }
  const features = [];
  for (const feature of geojson.features) {
    if (!feature || feature.type !== "Feature" || !feature.geometry) continue;
    const gType = feature.geometry.type;
    if (
      gType === "Point" ||
      gType === "LineString" ||
      gType === "Polygon" ||
      gType === "MultiPoint" ||
      gType === "MultiLineString" ||
      gType === "MultiPolygon"
    ) {
      features.push(feature);
    }
  }
  return { type: "FeatureCollection", features };
}

/**
 * Flatten nested Ariadne metadata and assign unique numeric ids so MapLibre
 * does not collapse comparison copies that share an OSM id.
 */
export function mapLibreCollection(collection) {
  const featuresIn = collection && Array.isArray(collection.features) ? collection.features : [];
  const labelToId = new Map();
  let nextOrdinal = 1;

  const assignTargetId = (props) => {
    const flat = flattenFeatureProperties(props);
    const explicit = targetIdFromProps(flat);
    const label = String(
      flat.analysis_target_label || flat.analysis_target || flat.target_label || "",
    ).trim();
    if (explicit) {
      if (label && !labelToId.has(label)) labelToId.set(label, explicit);
      const match = /^t(\d+)$/i.exec(explicit);
      if (match) {
        const ordinal = Number(match[1]);
        if (ordinal >= nextOrdinal) nextOrdinal = ordinal + 1;
      }
      return explicit;
    }
    if (!label) return "";
    if (!labelToId.has(label)) {
      labelToId.set(label, `t${nextOrdinal}`);
      nextOrdinal += 1;
    }
    return labelToId.get(label) || "";
  };

  return {
    type: "FeatureCollection",
    features: featuresIn.map((feature, index) => ({
      type: "Feature",
      id: index + 1,
      geometry: feature.geometry,
      properties: mapLibreProperties(feature.properties, assignTargetId(feature.properties)),
    })),
  };
}

export function flattenFeatureProperties(props) {
  const p = props && typeof props === "object" && !Array.isArray(props) ? { ...props } : {};
  const nested = p.ariadne && typeof p.ariadne === "object" && !Array.isArray(p.ariadne) ? p.ariadne : {};
  return { ...nested, ...p };
}

export function mapLibreProperties(props, assignedTargetId) {
  const p = flattenFeatureProperties(props);
  const tags = readTags(p);
  const targetId = assignedTargetId || targetIdFromProps(p);
  const shared = isSharedTarget(p);
  return {
    name: String(tags.name || p.name || ""),
    osm_id: p.osm_id != null ? String(p.osm_id) : "",
    osm_type: p.osm_type != null ? String(p.osm_type) : "",
    target_id: shared && !targetId ? "shared" : targetId,
    target_index: typeof p.target_index === "number" ? p.target_index : -1,
    target_label: String(p.target_label || p.analysis_target_label || p.analysis_target || ""),
    analysis_target: String(p.analysis_target || ""),
    analysis_target_label: String(
      p.analysis_target_label || p.analysis_target || p.target_label || "",
    ),
    primary_tag: primaryTag(tags) || "",
    tags_json: JSON.stringify(tags),
  };
}

function isSharedTarget(props) {
  const p = flattenFeatureProperties(props);
  const labels = [];
  if (Array.isArray(p.analysis_targets)) {
    for (const item of p.analysis_targets) {
      if (typeof item === "string" && item && !labels.includes(item)) labels.push(item);
    }
  }
  return labels.length > 1;
}

function fitToFeatures(collection) {
  if (!map || !collection.features.length) {
    // Preserve current view for empty results; do not call fitBounds.
    return;
  }
  const bounds = new maplibregl.LngLatBounds();
  let hasCoord = false;
  for (const feature of collection.features) {
    walkCoords(feature.geometry && feature.geometry.coordinates, (lng, lat) => {
      if (Number.isFinite(lng) && Number.isFinite(lat)) {
        bounds.extend([lng, lat]);
        hasCoord = true;
      }
    });
  }
  if (!hasCoord) return;
  map.fitBounds(bounds, { padding: 48, maxZoom: 15, duration: 600 });
}

function walkCoords(node, visit) {
  if (!Array.isArray(node)) return;
  if (node.length >= 2 && typeof node[0] === "number" && typeof node[1] === "number") {
    visit(node[0], node[1]);
    return;
  }
  for (const child of node) walkCoords(child, visit);
}

function popupHtml(feature) {
  // MapLibre requires HTML strings for popups; every interpolated value is escaped.
  const props = feature.properties || {};
  const tags = readTags(props);
  const rows = [
    ["Name", tags.name || props.name],
    ["Target", props.analysis_target_label || props.analysis_target || props.target_label],
    ["OSM type", props.osm_type],
    ["OSM ID", props.osm_id],
    ["Primary tag", props.primary_tag || primaryTag(tags)],
  ];
  const parts = ['<div class="map-popup">'];
  for (const [label, value] of rows) {
    if (value == null || value === "") continue;
    parts.push(
      `<div><strong>${escape(label)}:</strong> <span dir="auto">${escape(String(value))}</span></div>`,
    );
  }
  if (parts.length === 1) {
    parts.push("<div>No safe metadata available for this feature.</div>");
  }
  parts.push("</div>");
  return parts.join("");
}

function readTags(props) {
  const tags = props && props.tags;
  if (tags && typeof tags === "object" && !Array.isArray(tags)) return tags;
  const encoded = props && props.tags_json;
  if (typeof encoded === "string" && encoded) {
    try {
      const parsed = JSON.parse(encoded);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
    } catch {
      return {};
    }
  }
  return {};
}

function primaryTag(tags) {
  for (const key of ["leisure", "amenity", "landuse", "natural", "highway", "shop"]) {
    if (tags[key]) return `${key}=${tags[key]}`;
  }
  const keys = Object.keys(tags);
  if (!keys.length) return null;
  const key = keys[0];
  return `${key}=${tags[key]}`;
}

function escape(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** Test helper: classify a geometry for layer selection. */
export function layerForGeometryType(geometryType) {
  if (geometryType === "Point" || geometryType === "MultiPoint") return LAYER_POINT;
  if (geometryType === "LineString" || geometryType === "MultiLineString") return LAYER_LINE;
  if (geometryType === "Polygon" || geometryType === "MultiPolygon") return LAYER_FILL;
  return null;
}
