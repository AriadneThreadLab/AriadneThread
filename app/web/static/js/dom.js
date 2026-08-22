/**
 * Safe DOM helpers. Model output and feature properties are never trusted HTML.
 */

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function setText(el, value) {
  if (!el) return;
  el.textContent = value == null ? "" : String(value);
}

export function clearChildren(el) {
  if (!el) return;
  while (el.firstChild) {
    el.removeChild(el.firstChild);
  }
}

/** True only for http(s) URLs with no credentials or javascript: schemes. */
export function isSafeHttpUrl(value) {
  if (typeof value !== "string") return false;
  const trimmed = value.trim();
  if (!trimmed) return false;
  let url;
  try {
    url = new URL(trimmed);
  } catch {
    return false;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return false;
  if (url.username || url.password) return false;
  return true;
}

export function createExternalLink(href, label) {
  const a = document.createElement("a");
  a.href = href;
  a.textContent = label;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  return a;
}

export function appendLines(el, text) {
  clearChildren(el);
  const parts = String(text ?? "").split("\n");
  parts.forEach((line, index) => {
    el.appendChild(document.createTextNode(line));
    if (index < parts.length - 1) {
      el.appendChild(document.createElement("br"));
    }
  });
}
