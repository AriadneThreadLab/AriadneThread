/**
 * Same-origin client for POST /api/v1/agent/query
 */

export const AGENT_QUERY_PATH = "/api/v1/agent/query";
export const HEALTH_PATH = "/health";

const DEFAULT_TIMEOUT_MS = 240_000;

/**
 * @param {string} message
 * @param {{ signal?: AbortSignal, timeoutMs?: number, conversationId?: string }} [options]
 */
export async function queryAgent(message, options = {}) {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const external = options.signal;
  const onExternalAbort = () => controller.abort();
  if (external) {
    if (external.aborted) controller.abort();
    else external.addEventListener("abort", onExternalAbort, { once: true });
  }
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const payload = { message };
    if (options.conversationId) payload.conversation_id = options.conversationId;
    const response = await fetch(AGENT_QUERY_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    let body = null;
    const text = await response.text();
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        throw makeApiError(
          response.status,
          "The server returned a malformed JSON response.",
          null,
        );
      }
    }

    if (!response.ok) {
      throw makeApiError(response.status, messageFromErrorBody(body, response.status), body);
    }
    if (!body || typeof body !== "object") {
      throw makeApiError(500, "The server returned an empty response.", body);
    }
    return body;
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw makeApiError(504, "The request timed out or was cancelled.", null);
    }
    if (err && err.isApiError) throw err;
    throw makeApiError(0, "Network failure while contacting the agent API.", null);
  } finally {
    clearTimeout(timer);
    if (external) external.removeEventListener("abort", onExternalAbort);
  }
}

export async function fetchHealth() {
  const response = await fetch(HEALTH_PATH, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }
  return response.json();
}

function messageFromErrorBody(body, status) {
  if (body && typeof body.detail === "string" && body.detail.trim()) {
    return body.detail;
  }
  if (body && Array.isArray(body.detail)) {
    return "The request failed validation. Check that your message is not empty.";
  }
  switch (status) {
    case 422:
      return "The request failed validation.";
    case 429:
      return "The upstream service rate-limited this request. Please try again shortly.";
    case 503:
      return "A required service is temporarily unavailable.";
    case 504:
      return "The request timed out before a result was ready.";
    case 500:
      return "An unexpected server error occurred.";
    default:
      return `Request failed with status ${status}.`;
  }
}

function makeApiError(status, message, body) {
  const error = new Error(message);
  error.name = "ApiError";
  error.isApiError = true;
  error.status = status;
  error.body = body;
  return error;
}

/** Human-readable English label for an operational trace event. */
export function labelForTraceEvent(event, tool) {
  switch (event) {
    case "request_received":
      return "Request received";
    case "llm_turn":
      return "Model planning turn";
    case "memory_reuse":
      return "Execution memory decision";
    case "protocol_repair":
      return "Prompt format corrected";
    case "tool_call":
      if (tool === "search_osm_knowledge") return "Searched OSM documentation";
      if (tool === "query_osm") return "Queried live OpenStreetMap data";
      if (tool === "analyze_features") return "Requested spatial analytics";
      return tool ? `Requested tool: ${tool}` : "Tool call";
    case "tool_result":
      if (tool === "search_osm_knowledge") return "Received OSM documentation";
      if (tool === "query_osm") return "Normalized features to GeoJSON";
      if (tool === "analyze_features") return "Computed spatial analytics";
      return tool ? `Tool result: ${tool}` : "Tool result";
    case "tool_error":
      if (tool === "query_osm") return "Live OpenStreetMap query failed";
      return tool ? `Tool error: ${tool}` : "Tool error";
    case "final_answer":
      return "Model generated final response";
    case "stopped":
      return "Stopped";
    default:
      return event || "Operational event";
  }
}
