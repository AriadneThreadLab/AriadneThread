/**
 * Solar Decision Support Results panel.
 * Renders copied SolarDSS values only. Missing fields become an em dash.
 */

import { clearChildren, setText } from "./dom.js";

export const SOLAR_RESULT_FIELDS = [
  ["decision_mode", "Decision Mode"],
  ["project_type", "Project Type"],
  ["selected_site", "Selected Site"],
  ["suitability_score", "Suitability Score"],
  ["ghi", "GHI"],
  ["dni", "DNI"],
  ["slope", "Slope"],
  ["aspect", "Aspect"],
  ["shading_impact", "Shading Impact"],
  ["roof_suitability", "Roof Suitability"],
  ["available_area", "Available Area"],
  ["distance_to_grid", "Distance to Grid"],
  ["distance_to_road", "Distance to Road"],
  ["protected_area_status", "Protected Area Status"],
  ["recommended_capacity", "Recommended Capacity"],
  ["module_technology", "Module Technology"],
  ["tilt", "Tilt"],
  ["azimuth", "Azimuth"],
  ["expected_annual_energy", "Expected Annual Energy"],
  ["specific_yield", "Specific Yield"],
];

export const SOLAR_COMPARISON_COLUMNS = [
  ["site", "Site"],
  ["score", "Score"],
  ["ghi", "GHI"],
  ["slope", "Slope"],
  ["distance_to_grid", "Grid Distance"],
  ["annual_energy", "Annual Energy"],
];

let currentReport = null;
let currentRoot = null;
let selectedCandidateId = "";
let selectHandler = null;

export function renderSolarDecisionSupport(report, root, section, options) {
  currentRoot = root;
  currentReport = report && typeof report === "object" ? report : null;
  selectHandler = options && typeof options.onSelect === "function" ? options.onSelect : null;
  if (!root || !section) return 0;
  clearChildren(root);
  if (!currentReport) {
    section.hidden = true;
    selectedCandidateId = "";
    return 0;
  }
  section.hidden = false;

  const meta = document.createElement("dl");
  meta.className = "solar-meta energy-meta";
  for (const [key, label] of SOLAR_RESULT_FIELDS) {
    addDefinition(meta, label, formatSolarField(currentReport, key));
  }
  root.appendChild(meta);

  const inspected = document.createElement("section");
  inspected.className = "solar-inspected";
  const inspectedHeading = document.createElement("h3");
  setText(inspectedHeading, "Inspected candidate");
  inspected.appendChild(inspectedHeading);
  const inspectedBody = document.createElement("dl");
  inspectedBody.className = "solar-meta energy-meta";
  inspectedBody.id = "solar-inspected-body";
  inspected.appendChild(inspectedBody);
  root.appendChild(inspected);

  const candidates = Array.isArray(currentReport.candidates) ? currentReport.candidates : [];
  if (candidates.length) {
    root.appendChild(buildComparisonTable(candidates));
  }

  root.appendChild(buildProvenance(currentReport.provenance, currentReport.capability_id));
  const initialId =
    selectedCandidateId ||
    currentReport.selected_site ||
    (candidates[0] && (candidates[0].candidate_id || ""));
  if (initialId) selectSolarCandidate(initialId);
  return 1;
}

export function selectSolarCandidate(candidateId, featureProps) {
  selectedCandidateId = candidateId == null ? "" : String(candidateId);
  if (!currentRoot || !currentReport) return;
  const table = currentRoot.querySelector(".solar-comparison-table");
  if (table) {
    for (const row of table.querySelectorAll("tbody tr")) {
      row.classList.toggle("solar-row-selected", row.dataset.candidateId === selectedCandidateId);
    }
  }
  const inspected = currentRoot.querySelector("#solar-inspected-body");
  if (inspected) {
    clearChildren(inspected);
    const row = inspectedCandidate(selectedCandidateId, featureProps);
    addDefinition(inspected, "Site", row.site || "—");
    addDefinition(inspected, "Score", row.score);
    addDefinition(inspected, "GHI", row.ghi);
    addDefinition(inspected, "Slope", row.slope);
    addDefinition(inspected, "Grid Distance", row.distance_to_grid);
    addDefinition(inspected, "Annual Energy", row.annual_energy);
  }
}

function inspectedCandidate(candidateId, featureProps) {
  const candidates = Array.isArray(currentReport.candidates) ? currentReport.candidates : [];
  const fromReport = candidates.find((item) => String(item.candidate_id) === String(candidateId));
  const props = featureProps && typeof featureProps === "object" ? featureProps : {};
  const source = fromReport || {};
  return {
    site: source.name || source.candidate_id || props.name || props.candidate_id || candidateId || "—",
    score: formatSolarNumber(firstNumber(source.score, props.score)),
    ghi: formatSolarNumber(firstNumber(source.ghi, props.ghi), unitFor("ghi")),
    slope: formatSolarNumber(firstNumber(source.slope, props.slope), unitFor("slope")),
    distance_to_grid: formatSolarNumber(
      firstNumber(source.distance_to_grid, props.distance_to_grid),
      unitFor("distance_to_grid"),
    ),
    annual_energy: formatSolarNumber(
      firstNumber(source.annual_energy, props.expected_annual_energy, props.annual_energy),
      unitFor("expected_annual_energy"),
    ),
  };
}

function buildComparisonTable(candidates) {
  const wrap = document.createElement("div");
  wrap.className = "energy-table-wrap solar-table-wrap";
  const heading = document.createElement("h3");
  setText(heading, "Candidate comparison");
  wrap.appendChild(heading);
  const table = document.createElement("table");
  table.className = "energy-table solar-comparison-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const [, label] of SOLAR_COMPARISON_COLUMNS) {
    const th = document.createElement("th");
    setText(th, label);
    headRow.appendChild(th);
  }
  head.appendChild(headRow);
  table.appendChild(head);
  const body = document.createElement("tbody");
  for (const candidate of candidates) {
    const tr = document.createElement("tr");
    const id = String(candidate.candidate_id || "");
    tr.dataset.candidateId = id;
    tr.tabIndex = 0;
    tr.addEventListener("click", () => {
      selectSolarCandidate(id);
      if (selectHandler) selectHandler(id);
    });
    tr.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      selectSolarCandidate(id);
      if (selectHandler) selectHandler(id);
    });
    appendCell(tr, candidate.name || candidate.candidate_id || "—");
    appendCell(tr, formatSolarNumber(candidate.score));
    appendCell(tr, formatSolarNumber(candidate.ghi, unitFor("ghi")));
    appendCell(tr, formatSolarNumber(candidate.slope, unitFor("slope")));
    appendCell(tr, formatSolarNumber(candidate.distance_to_grid, unitFor("distance_to_grid")));
    appendCell(tr, formatSolarNumber(candidate.annual_energy, unitFor("expected_annual_energy")));
    body.appendChild(tr);
  }
  table.appendChild(body);
  wrap.appendChild(table);
  return wrap;
}

function buildProvenance(provenance, capabilityId) {
  const details = document.createElement("details");
  details.className = "tech-details solar-provenance";
  const summary = document.createElement("summary");
  setText(summary, "SolarDSS provenance");
  details.appendChild(summary);
  const list = document.createElement("dl");
  list.className = "solar-meta energy-meta";
  const source = provenance && typeof provenance === "object" ? provenance : {};
  addDefinition(list, "Engine", source.engine || "SolarDSS");
  addDefinition(list, "Source", source.source || "integration_layer");
  addDefinition(list, "Capability", source.capability_id || capabilityId);
  addDefinition(list, "Plugin", source.plugin_id);
  addDefinition(list, "SolarDSS version", source.solardss_version);
  addDefinition(list, "Request fingerprint", source.request_fingerprint);
  addDefinition(list, "Detail", source.detail);
  details.appendChild(list);
  return details;
}

function formatSolarField(report, key) {
  const value = report[key];
  if (value == null || value === "") return "—";
  if (typeof value === "number") return formatSolarNumber(value, unitFor(key));
  return String(value);
}

function unitFor(key) {
  const units = currentReport && currentReport.units && typeof currentReport.units === "object"
    ? currentReport.units
    : {};
  return typeof units[key] === "string" && units[key] ? units[key] : "";
}

export function formatSolarNumber(value, unit) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const text = Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
  return unit ? `${text} ${unit}` : text;
}

function firstNumber(...values) {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function addDefinition(list, label, value) {
  if (value == null || value === "") return;
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  setText(dt, label);
  setText(dd, String(value));
  list.appendChild(dt);
  list.appendChild(dd);
}

function appendCell(row, value) {
  const td = document.createElement("td");
  setText(td, value == null || value === "" ? "—" : String(value));
  row.appendChild(td);
}
