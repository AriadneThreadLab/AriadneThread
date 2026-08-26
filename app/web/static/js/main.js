/**
 * Ariadne Thread English demo UI controller.
 */

import { fetchHealth, labelForTraceEvent, queryAgent } from "./api.js";
import {
  appendLines,
  clearChildren,
  createExternalLink,
  isSafeHttpUrl,
  setText,
} from "./dom.js";
import { buildGeoJsonFilename, downloadGeoJson, GEOJSON_MIME, previewGeoJson } from "./download.js";
import {
  clearAnalysisArea,
  clearGeoJson,
  initMap,
  setAnalysisArea,
  setGeoJson,
} from "./map.js";

const EXAMPLE_QUERIES = [
  "Find public parks around Istanbul Technical University. Return at most 20 features.",
  "Which university has a larger proportion of green space within 2 km: Istanbul Technical University or Boğaziçi University?",
  "Compare road density around Istanbul Technical University and Boğaziçi University.",
  "Add Yıldız Technical University to the previous comparison.",
];

const CONVERSATION_STORAGE_KEY = "ariadne-conversation-id";
const PROTOCOL_LEAK_RE = /(tool_calls|final_answer|```|<think\b|reasoning_content)/i;

function conversationId() {
  try {
    let value = sessionStorage.getItem(CONVERSATION_STORAGE_KEY);
    if (!value) {
      value = crypto.randomUUID();
      sessionStorage.setItem(CONVERSATION_STORAGE_KEY, value);
    }
    return value;
  } catch {
    return undefined;
  }
}

function rememberConversationId(value) {
  if (!value) return;
  try {
    sessionStorage.setItem(CONVERSATION_STORAGE_KEY, value);
  } catch {
    /* ignore */
  }
}

const els = {};
let activeController = null;
let copyResetTimer = null;
let latestGeoJson = null;
let liveQueryExecuted = false;
let downloadResetTimer = null;

document.addEventListener("DOMContentLoaded", () => {
  bindElements();
  bindEvents();
  renderExamples();
  setUiState("idle");
  setRequestChip("Ready", "chip-idle");
  initMap("map");
  void refreshHealthOnce();
});

function bindElements() {
  const ids = [
    "query-form",
    "query-input",
    "submit-btn",
    "clear-btn",
    "examples",
    "status-banner",
    "health-dot",
    "health-label",
    "request-chip",
    "model-chip",
    "metrics-section",
    "metrics",
    "answer",
    "answer-empty",
    "answer-empty-text",
    "sources",
    "sources-empty",
    "sources-section",
    "workflow",
    "workflow-empty",
    "warnings",
    "warnings-section",
    "errors",
    "errors-section",
    "live-summary",
    "live-empty",
    "feature-count",
    "effective-limit",
    "scope-summary",
    "live-source",
    "warning-count",
    "live-error-row",
    "live-error",
    "map-message",
    "map-feature-chip",
    "map-scope-chip",
    "overpass-details",
    "overpass-query",
    "copy-query-btn",
    "download-geojson-btn",
    "download-status",
    "download-help",
    "geojson-details",
    "geojson-preview",
    "geojson-preview-note",
    "stop-reason",
    "analysis-section",
    "comparison-report",
    "analysis-method",
    "analysis-method-details",
    "analysis-provenance",
  ];
  for (const id of ids) {
    els[id] = document.getElementById(id);
  }
}

function bindEvents() {
  els["query-form"].addEventListener("submit", onSubmit);
  els["clear-btn"].addEventListener("click", onClear);
  els["copy-query-btn"].addEventListener("click", onCopyQuery);
  els["download-geojson-btn"].addEventListener("click", onDownloadGeoJson);
}

function renderExamples() {
  clearChildren(els.examples);
  for (const text of EXAMPLE_QUERIES) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "example-chip";
    btn.textContent = text;
    btn.addEventListener("click", () => {
      els["query-input"].value = text;
      els["query-input"].focus();
    });
    els.examples.appendChild(btn);
  }
}

async function refreshHealthOnce() {
  try {
    const health = await fetchHealth();
    els["health-dot"].dataset.state = health.status === "ok" ? "ok" : "bad";
    setText(
      els["health-label"],
      health.status === "ok" ? "Service online" : "Service status unknown",
    );
    if (health.llm_model) {
      const provider = health.llm_provider ? ` (${health.llm_provider})` : "";
      setText(els["model-chip"], `Model: ${health.llm_model}${provider}`);
    }
  } catch {
    els["health-dot"].dataset.state = "bad";
    setText(els["health-label"], "Service status unavailable");
  }
}

async function onSubmit(event) {
  event.preventDefault();
  if (activeController) return;

  const message = els["query-input"].value.trim();
  if (message.length < 2) {
    showBanner("error", "Enter a geographic question before submitting.");
    els["query-input"].focus();
    return;
  }

  activeController = new AbortController();
  setUiState("loading");
  setRequestChip("Processing", "chip-ai");
  showBanner("loading", "Analyzing your request and running geographic tools...");
  setBusy(true);
  resetResultPanels();
  clearGeoJson();
  clearAnalysisArea();
  setText(els["map-message"], "Waiting for live map data…");

  try {
    const result = await queryAgent(message, {
      signal: activeController.signal,
      conversationId: conversationId(),
    });
    rememberConversationId(result.conversation_id);
    renderResult(result);
  } catch (err) {
    renderRequestFailure(err);
  } finally {
    activeController = null;
    setBusy(false);
  }
}

function onClear() {
  if (activeController) {
    activeController.abort();
    activeController = null;
  }
  els["query-input"].value = "";
  resetResultPanels();
  clearGeoJson();
  clearAnalysisArea();
  setUiState("idle");
  setRequestChip("Ready", "chip-idle");
  showBanner("idle", "Ask a geographic question to begin.");
  setText(els["map-message"], "Submit a request to view live OpenStreetMap features.");
  els["query-input"].focus();
}

async function onCopyQuery() {
  const text = els["overpass-query"].textContent || "";
  if (!text.trim()) return;
  try {
    await navigator.clipboard.writeText(text);
    setText(els["copy-query-btn"], "Copied");
  } catch {
    setText(els["copy-query-btn"], "Copy failed");
  }
  if (copyResetTimer) clearTimeout(copyResetTimer);
  copyResetTimer = setTimeout(() => setText(els["copy-query-btn"], "Copy query"), 1600);
}

function onDownloadGeoJson() {
  if (!liveQueryExecuted || !latestGeoJson) {
    setText(els["download-status"], "Download GeoJSON is unavailable until a live query completes.");
    return;
  }
  setText(els["download-geojson-btn"], "Preparing download...");
  const result = downloadGeoJson(latestGeoJson, buildGeoJsonFilename(new Date()));
  if (result.ok) {
    setText(els["download-geojson-btn"], "Downloaded");
    setText(
      els["download-status"],
      `Downloaded ${result.filename} (${GEOJSON_MIME}).`,
    );
  } else {
    setText(els["download-geojson-btn"], "Download failed");
    setText(els["download-status"], result.error || "Download failed.");
  }
  if (downloadResetTimer) clearTimeout(downloadResetTimer);
  downloadResetTimer = setTimeout(() => {
    setText(els["download-geojson-btn"], "Download GeoJSON");
  }, 1800);
}

function resetResultPanels() {
  appendLines(els.answer, "");
  els["answer-empty"].hidden = false;
  setText(els["answer-empty-text"], "No answer yet.");
  clearChildren(els.sources);
  els["sources-empty"].hidden = false;
  if (els["sources-section"]) els["sources-section"].hidden = false;
  clearChildren(els.workflow);
  els["workflow-empty"].hidden = false;
  clearChildren(els.warnings);
  els["warnings-section"].hidden = true;
  clearChildren(els.errors);
  els["errors-section"].hidden = true;
  els["live-summary"].hidden = true;
  els["live-empty"].hidden = false;
  setText(els["feature-count"], "—");
  setText(els["effective-limit"], "—");
  setText(els["scope-summary"], "—");
  setText(els["warning-count"], "—");
  setText(els["live-source"], "—");
  setText(els["live-error"], "—");
  if (els["live-error-row"]) els["live-error-row"].hidden = true;
  setText(els["overpass-query"], "");
  els["overpass-details"].hidden = true;
  setText(els["geojson-preview"], "");
  els["geojson-details"].hidden = true;
  els["geojson-preview-note"].hidden = true;
  setText(els["stop-reason"], "");
  setText(els["copy-query-btn"], "Copy query");
  setText(els["download-geojson-btn"], "Download GeoJSON");
  setText(els["download-status"], "");
  setText(els["download-help"], "Enabled after a live Overpass result is available.");
  els["download-geojson-btn"].disabled = true;
  els["metrics-section"].hidden = true;
  clearChildren(els.metrics);
  setText(els["map-feature-chip"], "Features: 0");
  setText(els["map-scope-chip"], "Scope: —");
  setText(els["model-chip"], "Model: —");
  latestGeoJson = null;
  liveQueryExecuted = false;
  clearChildren(els["comparison-report"]);
  clearChildren(els["analysis-method"]);
  clearChildren(els["analysis-provenance"]);
  els["analysis-section"].hidden = true;
}

function renderResult(result) {
  const status = typeof result.status === "string" ? result.status : "";
  const answer = sanitizeAnswer(typeof result.answer === "string" ? result.answer : "", status);
  if (answer.trim()) {
    appendLines(els.answer, answer);
    els["answer-empty"].hidden = true;
    setText(els["answer-empty-text"], "");
  } else {
    appendLines(els.answer, "");
    els["answer-empty"].hidden = false;
    setText(els["answer-empty-text"], answerForStatus(status, result));
  }

  if (result.model) {
    setText(els["model-chip"], `Model: ${result.model}`);
  }

  const liveFailed = Boolean(result.live_query_failed);
  const liveDataAvailable = Boolean(
    result.live_data_available && result.geojson && result.geojson.type === "FeatureCollection",
  );
  const liveAttempted = Boolean(result.live_query_executed) || liveFailed || liveDataAvailable;

  renderSources(Array.isArray(result.knowledge_sources) ? result.knowledge_sources : []);
  const localRender = renderWorkflow(
    Array.isArray(result.execution_trace) ? result.execution_trace : [],
    liveDataAvailable,
  );
  renderAnalysis(result.analysis, result.execution_memory);
  renderList(els.warnings, els["warnings-section"], result.warnings, "warning");
  renderList(els.errors, els["errors-section"], result.errors, "error");
  renderMetrics(result, localRender);

  liveQueryExecuted = liveDataAvailable;
  const geojson = result.geojson;
  const featureCount =
    typeof result.feature_count === "number" ? result.feature_count : null;

  if (typeof result.overpass_query === "string" && result.overpass_query.trim()) {
    setText(els["overpass-query"], result.overpass_query);
    els["overpass-details"].hidden = false;
  }

  if (result.stop_reason) {
    setText(els["stop-reason"], `Stop reason: ${result.stop_reason}`);
  }

  applyStatusBanner(status, result);

  if (!liveAttempted) {
    clearGeoJson();
    clearAnalysisArea();
    latestGeoJson = null;
    els["download-geojson-btn"].disabled = true;
    els["live-summary"].hidden = true;
    els["live-empty"].hidden = false;
    setText(els["map-message"], "No live map data was requested for this question.");
    setText(els["map-feature-chip"], "Features: 0");
    setText(els["map-scope-chip"], "Scope: documentation");
    return;
  }

  els["live-summary"].hidden = false;
  els["live-empty"].hidden = true;
  setText(
    els["effective-limit"],
    typeof result.effective_limit === "number" ? String(result.effective_limit) : "—",
  );
  setText(els["scope-summary"], result.scope_summary || "—");
  setText(els["warning-count"], String((result.warnings || []).length));
  setText(
    els["live-source"],
    result.attribution || "© OpenStreetMap contributors (via Overpass API)",
  );
  setText(els["map-scope-chip"], `Scope: ${shortScope(result.scope_summary)}`);

  if (liveFailed || !liveDataAvailable) {
    clearGeoJson();
    clearAnalysisArea();
    latestGeoJson = null;
    els["download-geojson-btn"].disabled = true;
    setText(els["download-help"], "GeoJSON download is unavailable because no live result was returned.");
    setText(els["download-status"], "");
    setText(els["feature-count"], "—");
    setText(els["map-feature-chip"], "Features: —");
    els["geojson-details"].hidden = true;
    const liveError = liveErrorLabel(result);
    setText(els["live-error"], liveError);
    if (els["live-error-row"]) els["live-error-row"].hidden = false;
    if (status === "timed_out" || result.live_error_code === "overpass_timeout") {
      setText(els["map-message"], "The live OpenStreetMap request timed out.");
    } else if (status === "rate_limited") {
      setText(els["map-message"], "The Overpass service temporarily rate-limited this request.");
    } else {
      setText(els["map-message"], "Live OpenStreetMap data could not be loaded.");
    }
    return;
  }

  const collection = geojson;
  latestGeoJson = collection;
  setGeoJson(collection);
  updateGeoJsonPreview(collection);
  els["download-geojson-btn"].disabled = false;
  setText(els["download-help"], "Downloads the exact FeatureCollection returned by the API.");
  if (els["live-error-row"]) els["live-error-row"].hidden = true;

  const count = featureCount != null ? featureCount : (collection.features || []).length;
  setText(els["feature-count"], String(count));
  setText(els["map-feature-chip"], `Features: ${count}`);
  setAnalysisArea(trustedAnalysisScope(result));

  if (count === 0) {
    setText(
      els["map-message"],
      "The Overpass query completed, but no matching features were found.",
    );
  } else {
    setText(
      els["map-message"],
      "Displaying live OpenStreetMap features returned by the query.",
    );
  }
}

function liveErrorLabel(result) {
  const code = typeof result.live_error_code === "string" ? result.live_error_code : "";
  if (code === "overpass_timeout") return "Overpass timeout";
  if (code === "overpass_rate_limited") return "Overpass rate limited";
  if (code === "overpass_upstream_error") return "Overpass upstream error";
  if (code === "overpass_bad_response") return "Overpass bad response";
  if (code) return code;
  if (result.status === "timed_out") {
    const blob = ((result && result.errors) || []).join(" ").toLowerCase();
    if (blob.includes("llm_timeout")) return "Model planning timeout";
    return "Overpass timeout";
  }
  return "Live query failed";
}

function sanitizeAnswer(answer, status) {
  if (!answer.trim()) return "";
  if (PROTOCOL_LEAK_RE.test(answer) || answer.trim().startsWith("{")) {
    if (status === "invalid_model_response") {
      return (
        "The language model returned an invalid structured response. " +
        "No geographic query was executed."
      );
    }
    return "No final answer was produced because the model response was invalid.";
  }
  return answer;
}

function answerForStatus(status, result) {
  if (status === "invalid_model_response") {
    return (
      "No final answer was produced because the model response was invalid."
    );
  }
  if (status === "timed_out") {
    const blob = ((result && result.errors) || []).join(" ").toLowerCase();
    if (blob.includes("llm_timeout")) {
      return (
        "The local model timed out during request planning. " +
        "No live OSM query was executed."
      );
    }
    if (blob.includes("overpass") || result.live_query_failed) {
      return "The live OpenStreetMap query timed out.";
    }
    return "The request timed out before a result was ready.";
  }
  if (status === "no_matching_features") {
    return "The request completed, but no matching OpenStreetMap features were found.";
  }
  if (status === "failed" || status === "dependency_unavailable") {
    const blob = ((result && result.errors) || []).join(" ").toLowerCase();
    if (blob.includes("place_resolution") || blob.includes("place_ambiguous")) {
      return "One or more comparison locations could not be resolved reliably.";
    }
    if (blob.includes("llm_timeout")) {
      return (
        "The local model timed out during request planning. " +
        "No live OSM query was executed."
      );
    }
    return "The geographic query failed before live data could be retrieved.";
  }
  if (result && result.errors && result.errors.length) {
    const blob = result.errors.join(" ").toLowerCase();
    if (blob.includes("place_resolution") || blob.includes("place_ambiguous")) {
      return "One or more comparison locations could not be resolved reliably.";
    }
    if (blob.includes("llm_timeout")) {
      return (
        "The local model timed out during request planning. " +
        "No live OSM query was executed."
      );
    }
    return "The geographic query failed before live data could be retrieved.";
  }
  return "No answer yet.";
}

function applyStatusBanner(status, result) {
  const label = statusLabel(status);
  if (status === "invalid_model_response") {
    setRequestChip(label, "chip-fail");
    setUiState("failed");
    showBanner(
      "error",
      "Invalid model response. No geographic query was executed.",
    );
    return;
  }
  if (status === "timed_out") {
    setRequestChip(label, "chip-fail");
    setUiState("failed");
    showBanner("error", "Timed out before a complete result was ready.");
    return;
  }
  if (status === "rate_limited") {
    setRequestChip(label, "chip-warn");
    setUiState("partial");
    showBanner("warning", "The Overpass service temporarily rate-limited this request.");
    return;
  }
  if (status === "failed" || status === "dependency_unavailable") {
    setRequestChip(label, "chip-fail");
    setUiState("failed");
    showBanner("error", "Request failed. Review the Errors section.");
    return;
  }
  if (status === "no_matching_features") {
    setRequestChip(label, "chip-warn");
    setUiState("empty");
    showBanner("empty", "Live query completed with zero matching features.");
    return;
  }
  if (status === "completed_with_warnings" || (result.warnings && result.warnings.length)) {
    setRequestChip("Warning", "chip-warn");
    setUiState("partial");
    showBanner("warning", "Completed with warnings. Review the Warnings section.");
    return;
  }
  setRequestChip("Completed", "chip-ok");
  setUiState("success");
  showBanner(
    "success",
    result.live_query_executed ? "Live OpenStreetMap results ready." : "Documentation response ready.",
  );
}

function statusLabel(status) {
  switch (status) {
    case "completed":
      return "Completed";
    case "completed_with_warnings":
      return "Warning";
    case "no_matching_features":
      return "No matching features";
    case "failed":
      return "Request failed";
    case "timed_out":
      return "Request timed out";
    case "rate_limited":
      return "Rate limited";
    case "invalid_model_response":
      return "Invalid model response";
    case "dependency_unavailable":
      return "Request failed";
    default:
      return "Completed";
  }
}

function renderAnalysis(analysis, memory) {
  clearChildren(els["comparison-report"]);
  clearChildren(els["analysis-method"]);
  clearChildren(els["analysis-provenance"]);
  if (!analysis || typeof analysis !== "object") {
    els["analysis-section"].hidden = true;
    return;
  }
  els["analysis-section"].hidden = false;

  const report = analysis.report;
  if (report && Array.isArray(report.sections)) {
    for (const section of report.sections) {
      const block = document.createElement("article");
      block.className = "report-section";
      const heading = document.createElement("h3");
      setText(heading, section.title || section.key || "Section");
      block.appendChild(heading);
      const body = document.createElement("p");
      body.className = "report-section-body";
      setText(body, section.body || "");
      block.appendChild(body);
      els["comparison-report"].appendChild(block);
    }
  } else if (analysis.status && analysis.status !== "completed") {
    const note = document.createElement("p");
    setText(
      note,
      analysis.status === "abandoned"
        ? "Analytical re-planning was exhausted. No comparison report was produced."
        : "The analytical plan was rejected. See Analysis Method for details.",
    );
    els["comparison-report"].appendChild(note);
  }

  const trace = analysis.decision_trace || {};
  const selectedIndicator =
    trace.selected_indicator_id || trace.final_primary_metric || "—";
  const domain = trace.analysis_domain || "—";
  const candidates = Array.isArray(trace.candidate_indicators)
    ? trace.candidate_indicators.join(", ")
    : "—";
  const why = trace.indicator_selection_reason || (() => {
    const lines = [];
    for (const sel of Array.isArray(trace.metric_selections) ? trace.metric_selections : []) {
      if (sel.final_status !== "executed") continue;
      for (const evidence of Array.isArray(sel.evidence) ? sel.evidence : []) {
        if (evidence.statement) lines.push(evidence.statement);
      }
    }
    return lines.join(" ") || "—";
  })();
  const rules = [];
  for (const sel of Array.isArray(trace.metric_selections) ? trace.metric_selections : []) {
    if (sel.final_status !== "executed" && sel.final_status !== "superseded") continue;
    for (const evidence of Array.isArray(sel.evidence) ? sel.evidence : []) {
      if (evidence.rule_id && !rules.includes(evidence.rule_id)) {
        rules.push(evidence.rule_id);
      }
    }
  }

  const result = analysis.result;
  let scope = "—";
  let dataLine = "OpenStreetMap";
  const observationLines = [];
  if (result && Array.isArray(result.targets) && result.targets.length) {
    scope = result.targets.map((t) => t.data_provenance?.analysis_scope || t.label).join(" · ");
    const tags = result.targets[0].data_provenance?.resolved_tags || [];
    if (tags.length) dataLine = `OpenStreetMap — ${tags.join(", ")}`;
    for (const target of result.targets) {
      const primary = (target.metrics || []).find((m) => m.role === "primary");
      const n = primary ? primary.observation_count : 0;
      const missing = primary ? primary.missing_count : 0;
      const truncated =
        Boolean(target.data_provenance?.truncated) || Boolean(target.data_provenance?.limit_reached);
      observationLines.push(
        `${target.label}: ${target.data_provenance?.retrieved_feature_count ?? "—"} features (${n} observations, ${missing} missing)${
          truncated ? " — truncated at retrieval limit" : ""
        }`,
      );
    }
  }

  const methodRows = [
    ["Analysis Domain", domain],
    ["Candidate Indicators", candidates],
    ["Selected Indicator", selectedIndicator],
    ["Why Selected", typeof why === "string" ? why : "—"],
    ["Required Data", Array.isArray(trace.required_data) ? trace.required_data.join(", ") || "—" : "—"],
    [
      "OSM Grounding",
      Array.isArray(trace.osm_grounding) && trace.osm_grounding.length
        ? trace.osm_grounding.join("\n")
        : "—",
    ],
    ["Calculation Method", trace.calculation_method || "—"],
    ["Analysis Goal", trace.inferred_comparison_goal || analysis.plan?.comparison_goal || "—"],
    ["Selection Rules", rules.join(" · ") || "—"],
    ["Analysis Scope", scope],
    ["Data", dataLine],
    ["Observations", observationLines.join("\n") || "—"],
    [
      "Catalog / Ruleset",
      `${trace.indicator_catalog_version || trace.metric_catalog_version || "—"} · ${trace.ruleset_version || "—"}`,
    ],
  ];

  if (memory && memory.memory_reuse_attempted) {
    methodRows.push([
      "Follow-up",
      "This question revalidated the selected indicator against the current targets.",
    ]);
  }

  if (trace.plan_revision_count > 0) {
    const rejected = (trace.metric_selections || []).find(
      (s) => s.final_status === "rejected" || s.final_status === "superseded",
    );
    if (rejected) {
      methodRows.push([
        "Adjusted plan",
        `${rejected.metric} was not supported (${rejected.rejection_reason || "rejected"})` +
          (rejected.replacement_metric ? ` → ${rejected.replacement_metric}` : ""),
      ]);
    }
  }

  for (const [label, value] of methodRows) {
    const dt = document.createElement("dt");
    setText(dt, label);
    const dd = document.createElement("dd");
    setText(dd, value);
    els["analysis-method"].appendChild(dt);
    els["analysis-method"].appendChild(dd);
  }

  const prov = document.createElement("pre");
  prov.className = "code-block";
  setText(
    prov,
    JSON.stringify(
      {
        status: analysis.status,
        decision_trace: {
          analysis_domain: trace.analysis_domain || null,
          candidate_indicators: trace.candidate_indicators || [],
          selected_indicator_id: trace.selected_indicator_id || null,
          indicator_selection_reason: trace.indicator_selection_reason || null,
          required_data: trace.required_data || [],
          osm_grounding: trace.osm_grounding || [],
          calculation_method: trace.calculation_method || null,
          inferred_comparison_goal: trace.inferred_comparison_goal || null,
          final_primary_metric: trace.final_primary_metric || null,
          indicator_catalog_version: trace.indicator_catalog_version || null,
          metric_catalog_version: trace.metric_catalog_version || null,
          ruleset_version: trace.ruleset_version || null,
        },
        comparison: analysis.comparison || null,
      },
      null,
      2,
    ),
  );
  els["analysis-provenance"].appendChild(prov);
}

function renderSources(sources) {
  if (els["sources-section"]) els["sources-section"].hidden = false;
  clearChildren(els.sources);
  if (!sources.length) {
    els["sources-empty"].hidden = false;
    return;
  }
  els["sources-empty"].hidden = true;
  for (const source of sources) {
    const item = document.createElement("article");
    item.className = "source-card";

    const title = document.createElement("h3");
    title.className = "source-title";
    const titleText = source.title || "Untitled document";
    setText(title, titleText);
    title.title = titleText;
    item.appendChild(title);

    if (source.section) {
      const section = document.createElement("p");
      section.className = "source-section";
      setText(section, `Section: ${source.section}`);
      item.appendChild(section);
    }

    if (typeof source.score === "number") {
      const score = document.createElement("p");
      score.className = "source-score";
      setText(score, `Score: ${source.score.toFixed(3)}`);
      item.appendChild(score);
    }

    if (isSafeHttpUrl(source.url)) {
      const link = createExternalLink(source.url, source.url);
      link.className = "source-url";
      item.appendChild(link);
    } else {
      const bad = document.createElement("p");
      bad.className = "source-url-invalid";
      setText(bad, "Source URL omitted (not a safe http/https link).");
      item.appendChild(bad);
    }

    els.sources.appendChild(item);
  }
}

function renderWorkflow(steps, addLocalMapEvent) {
  clearChildren(els.workflow);
  if (!steps.length && !addLocalMapEvent) {
    els["workflow-empty"].hidden = false;
    return false;
  }
  els["workflow-empty"].hidden = true;
  let index = 0;
  for (const step of steps) {
    const event = String(step.event || "");
    const message = String(step.message || "");
    if (event.toLowerCase().includes("think") || message.includes("<think")) {
      continue;
    }
    if (PROTOCOL_LEAK_RE.test(message) && event !== "tool_call") {
      continue;
    }
    index += 1;
    // Memory decisions carry their own readable summary ("Metric validated: count").
    const label =
      event === "memory_reuse" && message ? message : labelForTraceEvent(event, step.tool);
    els.workflow.appendChild(
      workflowItem({
        index,
        label,
        status: step.status || "info",
        rows: workflowMetaRows(step),
        local: false,
      }),
    );
  }
  if (addLocalMapEvent) {
    index += 1;
    els.workflow.appendChild(
      workflowItem({
        index,
        label: "Rendered GeoJSON on the map",
        status: "completed",
        rows: [{ label: "note", value: "Client-side event (not a model decision)" }],
        local: true,
      }),
    );
    return true;
  }
  return false;
}

function workflowItem({ index, label, status, rows, local }) {
  const li = document.createElement("li");
  li.className = "workflow-card";
  li.dataset.status = status;
  if (local) li.dataset.local = "true";

  const row = document.createElement("div");
  row.className = "workflow-step";

  const num = document.createElement("span");
  num.className = "workflow-index";
  setText(num, String(index));
  row.appendChild(num);

  const title = document.createElement("span");
  title.className = "workflow-label";
  const titleText = String(label || "");
  setText(title, titleText);
  title.title = titleText;
  row.appendChild(title);
  li.appendChild(row);

  if (rows && rows.length) {
    const list = document.createElement("dl");
    list.className = "workflow-meta";
    for (const item of rows) {
      const block = document.createElement("div");
      block.className = "workflow-meta-row";
      const dt = document.createElement("dt");
      setText(dt, item.label);
      const dd = document.createElement("dd");
      setText(dd, item.value);
      dd.title = String(item.value);
      block.appendChild(dt);
      block.appendChild(dd);
      list.appendChild(block);
    }
    li.appendChild(list);
  }
  return li;
}

function workflowMetaRows(step) {
  const rows = [];
  const push = (label, value) => {
    if (value == null || value === "") return;
    const text = Array.isArray(value) ? value.join(", ") : String(value);
    if (!text) return;
    rows.push({ label, value: text });
  };
  push("status", step.status);
  push("tool", step.tool);
  push("error", step.error_code);
  const details = step.details && typeof step.details === "object" ? step.details : null;
  if (details) {
    const labels = {
      memory_step: "memory step",
      reused: "reused",
      recomputed: "recomputed",
      validated_tags: "validated tags",
      scope_type: "scope type",
      named_place: "named place",
      radius_meters: "radius meters",
      effective_limit: "effective limit",
      feature_count: "feature count",
      duration_ms: "duration ms",
      upstream_status: "upstream status",
      overpass_attempts: "overpass attempts",
      repair: "repair",
    };
    for (const key of Object.keys(labels)) {
      if (details[key] == null) continue;
      push(labels[key], details[key]);
    }
  }
  return rows;
}

function renderMetrics(result, localRender) {
  const tools = new Set(
    (result.execution_trace || [])
      .map((step) => step.tool)
      .filter((name) => typeof name === "string" && name),
  );
  const steps = (result.execution_trace || []).length + (localRender ? 1 : 0);
  const items = [
    ["Features", result.feature_count != null ? String(result.feature_count) : "—"],
    ["Tools used", String(tools.size)],
    ["Workflow steps", String(steps)],
    ["Warnings", String((result.warnings || []).length)],
    ["Stop reason", result.stop_reason || "—"],
    [
      "Effective result limit",
      typeof result.effective_limit === "number" ? String(result.effective_limit) : "—",
    ],
  ];
  clearChildren(els.metrics);
  for (const [label, value] of items) {
    const li = document.createElement("li");
    const strong = document.createElement("strong");
    setText(strong, label);
    li.appendChild(strong);
    li.appendChild(document.createTextNode(value));
    els.metrics.appendChild(li);
  }
  els["metrics-section"].hidden = false;
}

function updateGeoJsonPreview(collection) {
  const preview = previewGeoJson(collection);
  setText(els["geojson-preview"], preview.text);
  els["geojson-details"].hidden = false;
  els["geojson-preview-note"].hidden = !preview.truncated;
}

function trustedAnalysisScope(result) {
  // Only draw analysis geometry when backend scope summary indicates trusted
  // point/bbox data. Named-place queries stay textual.
  const summary = String(result.scope_summary || "");
  const point = summary.match(
    /Analysis area:\s*(\d+)\s*m around\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/i,
  );
  if (point) {
    return {
      type: "point",
      radius_m: Number(point[1]),
      lat: Number(point[2]),
      lon: Number(point[3]),
    };
  }
  const bbox = summary.match(
    /Bounding box:\s*south=(-?\d+(?:\.\d+)?),\s*west=(-?\d+(?:\.\d+)?),\s*north=(-?\d+(?:\.\d+)?),\s*east=(-?\d+(?:\.\d+)?)/i,
  );
  if (bbox) {
    return {
      type: "bbox",
      south: Number(bbox[1]),
      west: Number(bbox[2]),
      north: Number(bbox[3]),
      east: Number(bbox[4]),
    };
  }
  return null;
}

function shortScope(summary) {
  if (!summary) return "—";
  if (summary.length <= 42) return summary;
  return `${summary.slice(0, 39)}…`;
}

function renderList(container, section, values, kind) {
  clearChildren(container);
  const items = Array.isArray(values) ? values.filter((v) => typeof v === "string" && v.trim()) : [];
  if (!items.length) {
    section.hidden = true;
    return;
  }
  section.hidden = false;
  const ul = document.createElement("ul");
  ul.className = `${kind}-list`;
  for (const value of items) {
    const li = document.createElement("li");
    setText(li, value);
    ul.appendChild(li);
  }
  container.appendChild(ul);
}

function renderRequestFailure(err) {
  resetResultPanels();
  clearGeoJson();
  clearAnalysisArea();
  const message =
    err && typeof err.message === "string" && err.message
      ? err.message
      : "The request failed.";
  setUiState("failed");
  setRequestChip("Request failed", "chip-fail");
  showBanner("error", message);
  els["errors-section"].hidden = false;
  const ul = document.createElement("ul");
  ul.className = "error-list";
  const li = document.createElement("li");
  setText(li, message);
  ul.appendChild(li);
  els.errors.appendChild(ul);
  setText(els["map-message"], "Live OpenStreetMap data could not be loaded.");
  setText(
    els["answer-empty-text"],
    "The geographic query failed before live data could be retrieved.",
  );
  els["answer-empty"].hidden = false;
}

function setBusy(busy) {
  els["submit-btn"].disabled = busy;
  els["query-input"].disabled = busy;
  els["submit-btn"].textContent = busy ? "Working…" : "Submit question";
}

function setUiState(state) {
  document.body.dataset.uiState = state;
}

function setRequestChip(text, className) {
  els["request-chip"].className = `chip ${className}`;
  setText(els["request-chip"], text);
}

function showBanner(kind, message) {
  els["status-banner"].dataset.kind = kind;
  setText(els["status-banner"], message);
}
