/**
 * Browser-local GeoJSON download (no backend storage).
 */

export const GEOJSON_MIME = "application/geo+json";

/**
 * @param {object} geojson
 * @param {string} [filename]
 * @returns {{ ok: boolean, filename?: string, error?: string }}
 */
export function downloadGeoJson(geojson, filename) {
  if (!geojson || geojson.type !== "FeatureCollection" || !Array.isArray(geojson.features)) {
    return { ok: false, error: "No valid FeatureCollection is available to download." };
  }
  const name = filename || buildGeoJsonFilename(new Date());
  try {
    const text = JSON.stringify(geojson, null, 2);
    const blob = new Blob([text], { type: GEOJSON_MIME });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    return { ok: true, filename: name };
  } catch (err) {
    return {
      ok: false,
      error: err && err.message ? String(err.message) : "Download failed.",
    };
  }
}

export function buildGeoJsonFilename(date = new Date()) {
  const stamp = [
    date.getUTCFullYear(),
    String(date.getUTCMonth() + 1).padStart(2, "0"),
    String(date.getUTCDate()).padStart(2, "0"),
    "T",
    String(date.getUTCHours()).padStart(2, "0"),
    String(date.getUTCMinutes()).padStart(2, "0"),
    String(date.getUTCSeconds()).padStart(2, "0"),
  ].join("");
  return `ariadne-thread-osm-results-${stamp}.geojson`;
}

/** Truncate only the visual preview; downloads stay complete. */
export function previewGeoJson(geojson, maxChars = 12000) {
  const text = JSON.stringify(geojson, null, 2);
  if (text.length <= maxChars) {
    return { text, truncated: false };
  }
  return {
    text: `${text.slice(0, maxChars)}\n…`,
    truncated: true,
  };
}
