/**
 * Generic analysis-chart renderer. Renders bar, line, and scatter specs.
 * Does not know about GeoLoadST or any scientific engine.
 */

const CHART_TYPES = new Set(["bar", "line", "scatter"]);
const SERIES_COLORS = ["#0e7490", "#c026d3", "#b45309", "#1d4ed8"];

export function renderAnalysisCharts(charts, root, section) {
  if (!root) return 0;
  root.replaceChildren();
  const valid = Array.isArray(charts) ? charts.filter(isRenderableChart) : [];
  if (section) section.hidden = valid.length === 0;
  for (const spec of valid) {
    root.appendChild(renderAnalysisChart(spec));
  }
  return valid.length;
}

export function isRenderableChart(spec) {
  if (!spec || typeof spec !== "object") return false;
  if (!CHART_TYPES.has(spec.chart_type)) return false;
  if (typeof spec.title !== "string" || !spec.title.trim()) return false;
  const series = usableSeries(spec.series);
  return series.length > 0;
}

export function renderAnalysisChart(spec) {
  const article = document.createElement("article");
  article.className = "chart-card";
  article.dataset.chartId = String(spec.chart_id || "");
  article.dataset.chartType = String(spec.chart_type || "");

  const heading = document.createElement("h3");
  heading.textContent = String(spec.title || "Chart");
  article.appendChild(heading);

  const figure = document.createElement("figure");
  figure.className = "chart-figure";
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  tooltip.hidden = true;
  const svg = buildChartSvg(spec, tooltip);
  figure.appendChild(svg);
  figure.appendChild(tooltip);
  article.appendChild(figure);

  const axis = document.createElement("p");
  axis.className = "chart-axis-caption";
  const labels = [spec.x_label, spec.y_label].filter((item) => typeof item === "string" && item);
  axis.textContent = labels.join(" · ");
  if (axis.textContent) article.appendChild(axis);

  const provenance = provenanceText(spec.metadata);
  if (provenance) {
    const note = document.createElement("p");
    note.className = "chart-provenance";
    note.textContent = provenance;
    article.appendChild(note);
  }
  return article;
}

function usableSeries(raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  for (const series of raw) {
    if (!series || typeof series !== "object") continue;
    const name = typeof series.name === "string" && series.name.trim() ? series.name.trim() : "Series";
    const data = Array.isArray(series.data) ? series.data.filter(isPoint) : [];
    if (data.length) out.push({ name, data });
  }
  return out;
}

function isPoint(point) {
  if (!point || typeof point !== "object") return false;
  if (point.x == null || point.x === "") return false;
  return typeof point.y === "number" && Number.isFinite(point.y);
}

function buildChartSvg(spec, tooltip) {
  const width = 400;
  const height = 240;
  const pad = { top: 16, right: 16, bottom: 44, left: 48 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const series = usableSeries(spec.series);
  const categories = categoryLabels(series);
  const yMax = Math.max(0, ...series.flatMap((item) => item.data.map((point) => point.y)));
  const yMin = Math.min(0, ...series.flatMap((item) => item.data.map((point) => point.y)));
  const span = yMax - yMin || 1;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": String(spec.title || "Chart"),
  });
  svg.appendChild(
    svgEl("rect", { x: 0, y: 0, width, height, class: "chart-bg" }),
  );
  const plot = svgEl("g", { transform: `translate(${pad.left} ${pad.top})` });
  svg.appendChild(plot);

  const ticks = 4;
  for (let i = 0; i <= ticks; i += 1) {
    const ratio = i / ticks;
    const y = innerH - ratio * innerH;
    const value = yMin + span * ratio;
    plot.appendChild(
      svgEl("line", {
        x1: 0,
        y1: y,
        x2: innerW,
        y2: y,
        class: "chart-grid",
      }),
    );
    const label = svgEl("text", {
      x: -8,
      y,
      class: "chart-tick chart-tick-y",
    });
    label.textContent = formatTick(value);
    plot.appendChild(label);
  }

  plot.appendChild(svgEl("line", { x1: 0, y1: innerH, x2: innerW, y2: innerH, class: "chart-axis" }));
  plot.appendChild(svgEl("line", { x1: 0, y1: 0, x2: 0, y2: innerH, class: "chart-axis" }));

  if (spec.chart_type === "bar") {
    drawBars(plot, series, categories, innerW, innerH, yMin, span, tooltip);
  } else {
    drawLineOrScatter(plot, spec.chart_type, series, categories, innerW, innerH, yMin, span, tooltip);
  }

  categories.forEach((label, index) => {
    const x = ((index + 0.5) / Math.max(categories.length, 1)) * innerW;
    const text = svgEl("text", { x, y: innerH + 18, class: "chart-tick chart-tick-x" });
    text.textContent = String(label);
    plot.appendChild(text);
  });
  return svg;
}

function drawBars(plot, series, categories, innerW, innerH, yMin, span, tooltip) {
  const groupWidth = innerW / Math.max(categories.length, 1);
  const barWidth = Math.max(6, (groupWidth * 0.7) / Math.max(series.length, 1));
  categories.forEach((label, catIndex) => {
    series.forEach((item, seriesIndex) => {
      const point = item.data.find((entry) => String(entry.x) === String(label));
      if (!point) return;
      const h = ((point.y - yMin) / span) * innerH;
      const x = catIndex * groupWidth + (groupWidth - series.length * barWidth) / 2 + seriesIndex * barWidth;
      const y = innerH - h;
      const rect = svgEl("rect", {
        x,
        y,
        width: barWidth,
        height: Math.max(h, 0),
        class: "chart-bar",
        fill: SERIES_COLORS[seriesIndex % SERIES_COLORS.length],
      });
      bindTooltip(rect, tooltip, item.name, label, point.y);
      plot.appendChild(rect);
    });
  });
}

function drawLineOrScatter(plot, chartType, series, categories, innerW, innerH, yMin, span, tooltip) {
  const indexOf = new Map(categories.map((label, index) => [String(label), index]));
  series.forEach((item, seriesIndex) => {
    const color = SERIES_COLORS[seriesIndex % SERIES_COLORS.length];
    const coords = item.data
      .map((point) => {
        const idx = indexOf.get(String(point.x));
        if (idx == null) return null;
        const x = ((idx + 0.5) / Math.max(categories.length, 1)) * innerW;
        const y = innerH - ((point.y - yMin) / span) * innerH;
        return { x, y, label: point.x, value: point.y };
      })
      .filter(Boolean);
    if (chartType === "line" && coords.length > 1) {
      const line = svgEl("polyline", {
        points: coords.map((pt) => `${pt.x},${pt.y}`).join(" "),
        class: "chart-line",
        fill: "none",
        stroke: color,
      });
      plot.appendChild(line);
    }
    for (const pt of coords) {
      const dot = svgEl("circle", {
        cx: pt.x,
        cy: pt.y,
        r: 4,
        class: "chart-point",
        fill: color,
      });
      bindTooltip(dot, tooltip, item.name, pt.label, pt.value);
      plot.appendChild(dot);
    }
  });
}

function categoryLabels(series) {
  const labels = [];
  for (const item of series) {
    for (const point of item.data) {
      const key = String(point.x);
      if (!labels.includes(key)) labels.push(key);
    }
  }
  return labels;
}

function provenanceText(metadata) {
  if (!metadata || typeof metadata !== "object") return "";
  const parts = [];
  if (metadata.engine) parts.push(String(metadata.engine));
  if (metadata.capability_id || metadata.analysis) {
    parts.push(String(metadata.capability_id || metadata.analysis));
  }
  if (metadata.network_id) parts.push(String(metadata.network_id));
  const params = metadata.parameters && typeof metadata.parameters === "object" ? metadata.parameters : {};
  const paramParts = Object.keys(params).map((key) => `${key}=${params[key]}`);
  if (paramParts.length) parts.push(paramParts.join(", "));
  return parts.join(" · ");
}

function bindTooltip(node, tooltip, seriesName, x, y) {
  const text = `${seriesName}: ${x} = ${formatTick(y)}`;
  node.addEventListener("pointerenter", (event) => {
    tooltip.hidden = false;
    tooltip.textContent = text;
    positionTooltip(tooltip, event);
  });
  node.addEventListener("pointermove", (event) => positionTooltip(tooltip, event));
  node.addEventListener("pointerleave", () => {
    tooltip.hidden = true;
  });
}

function positionTooltip(tooltip, event) {
  const host = tooltip.parentElement;
  if (!host) return;
  const box = host.getBoundingClientRect();
  tooltip.style.left = `${Math.max(8, event.clientX - box.left + 8)}px`;
  tooltip.style.top = `${Math.max(8, event.clientY - box.top - 28)}px`;
}

function formatTick(value) {
  if (!Number.isFinite(value)) return "";
  if (Number.isInteger(value)) return String(value);
  return String(Number(value.toPrecision(4)));
}

function svgEl(name, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, String(value));
  }
  return node;
}
