"""Shared builders for offline active-learning tests.

No database, no network, no model: every collaborator here is a hand-written
fake or a plain domain object.
"""

from __future__ import annotations

from typing import Any

from app.agent.contracts import GeoAgentResponse, TraceEvent

COMPARISON_QUERY = (
    "Compare public parks within 2 km of University of Tehran and Sharif University of Technology."
)

ANALYSIS_PLAN: dict[str, Any] = {
    "analysis_type": "comparison",
    "feature_concept": "public parks",
    "comparison_goal": "which campus neighbourhood has more parks",
    "targets": [
        {
            "target_id": "ut",
            "label": "University of Tehran",
            "dataset_ref": "osm_result_1",
            "reference_points": [],
        },
        {
            "target_id": "sharif",
            "label": "Sharif University of Technology",
            "dataset_ref": "osm_result_2",
            "reference_points": [],
        },
    ],
    "metrics": [
        {
            "metric": "count",
            "role": "primary",
            "inferred_goal": "abundance",
            "claimed_rule_ids": ["ABUNDANCE_COUNT_001"],
            "user_explicit": False,
            "property_key": None,
            "direction": "higher_is_better",
            "ratio": None,
        }
    ],
}

COMPARISON_PLAN: dict[str, Any] = {
    "analysis_type": "comparison",
    "feature_concept": "public parks",
    "targets": [
        {"label": "University of Tehran", "place_query": "University of Tehran, Tehran, Iran"},
        {
            "label": "Sharif University of Technology",
            "place_query": "Sharif University of Technology, Tehran, Iran",
        },
    ],
    "radius_m": 2000,
    "comparison_goal": "which campus neighbourhood has more parks",
}


def trace(
    kind: str,
    message: str,
    *,
    tool: str | None = None,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> TraceEvent:
    return TraceEvent(
        kind=kind,  # type: ignore[arg-type]
        message=message,
        tool_name=tool,
        error_code=error_code,
        details=details,
    )


def successful_run(
    *,
    events: list[TraceEvent] | None = None,
    feature_count: int = 12,
) -> GeoAgentResponse:
    """A clean live search: grounded, resolved, queried, answered."""
    return GeoAgentResponse(
        answer="Found 12 parks near the requested campuses.",
        trace=events
        or [
            trace("tool_call", "calling search_osm_knowledge", tool="search_osm_knowledge"),
            trace(
                "tool_result",
                "received 5 documentation passage(s)",
                tool="search_osm_knowledge",
            ),
            trace("tool_call", "calling resolve_place", tool="resolve_place"),
            trace(
                "tool_result",
                "Resolved comparison target target=University of Tehran "
                "source=nominatim status=completed",
                tool="resolve_place",
            ),
            trace("tool_call", "calling query_osm", tool="query_osm"),
            trace("tool_result", "received 12 feature(s)", tool="query_osm"),
        ],
        feature_count=feature_count,
        overpass_query="[out:json];node[leisure=park];out 12;",
        scope_summary="within 2000 m of University of Tehran",
        validated_tags=["leisure=park"],
        stop_reason="final_answer",
        model="deepseek-r1:7b",
    )


def failing_run(
    *,
    error_code: str,
    message: str,
    tool: str = "query_osm",
) -> GeoAgentResponse:
    """A run whose failure is described by one accumulated error entry."""
    return GeoAgentResponse(
        answer="The request could not be completed.",
        trace=[
            trace("tool_call", f"calling {tool}", tool=tool),
            trace("tool_error", message, tool=tool, error_code=error_code),
        ],
        errors=[f"{error_code}: {message}"],
        stop_reason="final_answer",
        model="deepseek-r1:7b",
    )
