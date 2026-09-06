"""Request-scoped tool context (created once per agent run)."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.analytics.contracts import MetricSelectionTrace
from app.analytics.datasets import DatasetRegistry
from app.places.contracts import PlaceRegistry


@dataclass(slots=True)
class AnalysisRunState:
    """Mutable per-run analytics re-plan state."""

    plan_revision_count: int = 0
    accepted_traces: list[MetricSelectionTrace] = field(default_factory=list)
    rejected_traces: list[MetricSelectionTrace] = field(default_factory=list)
    last_plan_primary: str | None = None
    energy_network_id: str | None = None
    energy_capability_requested: str | None = None
    energy_capability_canonical: str | None = None


@dataclass(slots=True)
class GroundingState:
    """Mutable per-run semantic tag grounding (from documentation or first query)."""

    tags: list[str] | None = None


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Everything a tool may know about the current request."""

    datasets: DatasetRegistry
    analysis: AnalysisRunState
    user_message: str
    places: PlaceRegistry
    grounding: GroundingState
