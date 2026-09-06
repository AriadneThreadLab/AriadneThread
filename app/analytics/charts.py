"""Generic structured chart results for analytical responses.

Charts are deterministic data copied from an analysis engine. This module
validates the public shape. It does not compute scientific values.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ChartType = Literal["bar", "line", "scatter"]
SUPPORTED_CHART_TYPES: frozenset[str] = frozenset({"bar", "line", "scatter"})


class ChartPoint(BaseModel):
    """One plotted observation. ``y`` must already exist in the engine output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: str | int | float
    y: float

    @field_validator("y")
    @classmethod
    def _finite_y(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("chart point y must be a finite number")
        return float(value)

    @field_validator("x")
    @classmethod
    def _usable_x(cls, value: str | int | float) -> str | int | float:
        if isinstance(value, bool):
            raise ValueError("chart point x must not be a boolean")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("chart point x must be finite")
        if isinstance(value, str) and not value.strip():
            raise ValueError("chart point x must not be empty")
        return value


class ChartSeries(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    data: list[ChartPoint] = Field(min_length=1)


class ChartMetadata(BaseModel):
    """Provenance only. Never model reasoning."""

    model_config = ConfigDict(frozen=True)

    analysis: str | None = None
    engine: str | None = None
    capability_id: str | None = None
    network_id: str | None = None
    dataset_id: str | None = None
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class AnalysisChart(BaseModel):
    """One chart the UI can render. Unknown types are rejected by the parser."""

    model_config = ConfigDict(frozen=True)

    chart_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=160)
    chart_type: ChartType
    x_label: str = ""
    y_label: str = ""
    series: list[ChartSeries] = Field(min_length=1)
    metadata: ChartMetadata = Field(default_factory=ChartMetadata)


def parse_charts(raw: object) -> list[AnalysisChart]:
    """Keep well-formed charts. Drop malformed items instead of failing the run."""
    if not isinstance(raw, list | tuple):
        return []
    charts: list[AnalysisChart] = []
    seen: set[str] = set()
    for item in raw:
        chart = _as_chart(item)
        if chart is None or chart.chart_id in seen:
            continue
        seen.add(chart.chart_id)
        charts.append(chart)
    return charts


def _as_chart(raw: object) -> AnalysisChart | None:
    if isinstance(raw, AnalysisChart):
        return raw if raw.series else None
    if not isinstance(raw, dict):
        return None
    try:
        chart = AnalysisChart.model_validate(raw)
    except Exception:
        return None
    return chart


def bar_chart(
    *,
    chart_id: str,
    title: str,
    points: list[tuple[str | int | float, float]],
    x_label: str,
    y_label: str,
    series_name: str,
    metadata: ChartMetadata | None = None,
) -> AnalysisChart | None:
    """Build a bar chart from already-computed (x, y) pairs. Empty input is omitted."""
    series = _series(series_name, points)
    if series is None:
        return None
    return AnalysisChart(
        chart_id=chart_id,
        title=title,
        chart_type="bar",
        x_label=x_label,
        y_label=y_label,
        series=[series],
        metadata=metadata or ChartMetadata(),
    )


def line_chart(
    *,
    chart_id: str,
    title: str,
    points: list[tuple[str | int | float, float]],
    x_label: str,
    y_label: str,
    series_name: str,
    metadata: ChartMetadata | None = None,
) -> AnalysisChart | None:
    series = _series(series_name, points)
    if series is None:
        return None
    return AnalysisChart(
        chart_id=chart_id,
        title=title,
        chart_type="line",
        x_label=x_label,
        y_label=y_label,
        series=[series],
        metadata=metadata or ChartMetadata(),
    )


def _series(name: str, points: list[tuple[str | int | float, float]]) -> ChartSeries | None:
    data: list[ChartPoint] = []
    for raw_x, raw_y in points:
        try:
            data.append(ChartPoint(x=raw_x, y=raw_y))
        except Exception:
            continue
    if not data or not name.strip():
        return None
    return ChartSeries(name=name.strip(), data=data)


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        number = float(value)
        if math.isfinite(number):
            return number
    return None
