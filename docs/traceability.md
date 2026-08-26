# Traceability and explainability

Every Ariadne Thread result is meant to be followed from the question to the
final report. The system records an operational thread, not hidden
chain-of-thought.

## What is recorded

| Artifact | Role |
|---|---|
| `execution_trace` | Ordered workflow steps: request, planning turns, tool calls, results, stop reason |
| `knowledge_sources` | OSM Wiki citations used for grounding (title, section, URL, score) |
| `overpass_query` | The compiled Overpass QL for the live retrieval (read-only provenance) |
| `analysis.plan` | The validated analytical plan the tools executed |
| `analysis.decision_trace` | Domain, candidate indicators, selected indicator, method, and rules |
| `analysis.result` / `comparison` | Computed values and the comparison statement |
| Attribution | OpenStreetMap ODbL credit on live results |

Hidden `<think>` content is stripped at the LLM provider boundary. It is never
written to the trace, logs, or the user interface.

## Workflow the user can inspect

The Analysis Workflow panel renders `execution_trace` as compact steps:
tool name, status, scope, tags, limits, and feature counts. A client-side
“rendered GeoJSON on the map” event is marked as local and is not a model
decision.

OSM Documentation Sources lists the retrieved Wiki pages. If a follow-up reuses
previously validated grounding, those citations stay visible. The Comparison
Report states the analysis goal, selected indicator, why it was selected,
per-target values, ranking, units, and data limitations.

## Provenance on the map

Live features keep OSM identifiers and tags. Comparison layers are coloured by
analysis target. The downloaded GeoJSON is the same FeatureCollection the API
returned.

## Analytical thread

```
question → domain → catalog candidates → selected indicator
        → OSM data plan → grounded tags → place resolution → Overpass retrieval
        → deterministic indicator execution → comparison → report + map + citations
```

If a step fails (invalid plan, empty Overpass result, place ambiguity), the
trace still records the failure. Empty geographic results stay empty.
