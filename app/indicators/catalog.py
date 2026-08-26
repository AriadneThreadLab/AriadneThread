"""Indicator Catalog: load YAML definitions, validate, query.

The catalog is the only place an ``indicator_id`` becomes real. Unknown ids
are rejected here so a later tool schema can expose a closed enum.

This module does not talk to the agent, Overpass, or the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import yaml
from pydantic import ValidationError

from app.analytics.contracts import AnalysisGoal
from app.analytics.engine import SpatialAnalyticsEngine
from app.analytics.methods import METHOD_REGISTRY, MethodId, MethodSpec
from app.core.errors import IndicatorCatalogError, UnknownIndicatorError
from app.indicators.contracts import (
    DomainId,
    FeatureGeometry,
    IndicatorDefinition,
    RequirementKind,
)
from app.rag.corpus import is_allowed_source

INDICATOR_CATALOG_VERSION = "indicator-catalog-1"

_PACKAGE_DEFINITIONS = Path(__file__).resolve().parent / "definitions"
_TRIVIAL_METHODS: frozenset[MethodId] = frozenset({"count", "density", "statistic"})


@dataclass(frozen=True, slots=True)
class AvailableCapabilities:
    """Runtime capabilities used to filter catalog candidates.

    ``None`` on a field means "do not filter on this axis". Phase 1 uses this
    for catalog queries only; the agent is not wired yet.
    """

    method_ids: frozenset[MethodId] | None = None
    implemented_only: bool = False
    geometries: frozenset[FeatureGeometry] | None = None
    requirement_kinds: frozenset[RequirementKind] | None = None


class IndicatorCatalog:
    """Immutable collection of validated indicator definitions."""

    def __init__(
        self,
        definitions: tuple[IndicatorDefinition, ...],
        *,
        version: str = INDICATOR_CATALOG_VERSION,
    ) -> None:
        by_id: dict[str, IndicatorDefinition] = {}
        for definition in definitions:
            if definition.indicator_id in by_id:
                raise IndicatorCatalogError(f"duplicate indicator_id '{definition.indicator_id}'")
            by_id[definition.indicator_id] = definition
        self._by_id = by_id
        self._order = tuple(definition.indicator_id for definition in definitions)
        self.version = version

    def get_indicator(self, indicator_id: str) -> IndicatorDefinition:
        """Return the definition for ``indicator_id``.

        Raises:
            UnknownIndicatorError: if the id is not in the catalog.
        """
        try:
            return self._by_id[indicator_id]
        except KeyError as exc:
            known = ", ".join(self._order) or "none"
            raise UnknownIndicatorError(
                f"unknown indicator_id '{indicator_id}'; available: {known}"
            ) from exc

    def list_indicators(self) -> tuple[IndicatorDefinition, ...]:
        """All definitions in catalog-file order."""
        return tuple(self._by_id[indicator_id] for indicator_id in self._order)

    def list_by_domain(self, domain: DomainId) -> tuple[IndicatorDefinition, ...]:
        """Definitions whose ``domain`` matches, preserving catalog order."""
        _require_domain(domain)
        return tuple(item for item in self.list_indicators() if item.domain == domain)

    def find_candidates(
        self,
        domain: DomainId,
        analytical_goal: AnalysisGoal,
        available_capabilities: AvailableCapabilities | None = None,
    ) -> tuple[IndicatorDefinition, ...]:
        """Deterministic candidate list for one domain and analysis goal.

        Deprecated indicators are never returned. Filtering is exact (no
        embeddings, no ranking model).
        """
        _require_domain(domain)
        capabilities = available_capabilities or AvailableCapabilities()
        selected: list[IndicatorDefinition] = []
        for definition in self.list_by_domain(domain):
            if definition.status == "deprecated":
                continue
            if analytical_goal not in definition.goals:
                continue
            if not _matches_capabilities(definition, capabilities):
                continue
            selected.append(definition)
        return tuple(selected)

    def indicator_ids(self) -> tuple[str, ...]:
        return self._order


def get_indicator(indicator_id: str) -> IndicatorDefinition:
    """Look up an indicator in the default catalog."""
    return INDICATOR_CATALOG.get_indicator(indicator_id)


def list_indicators() -> tuple[IndicatorDefinition, ...]:
    """List every indicator in the default catalog."""
    return INDICATOR_CATALOG.list_indicators()


def list_by_domain(domain: DomainId) -> tuple[IndicatorDefinition, ...]:
    """List default-catalog indicators for one domain."""
    return INDICATOR_CATALOG.list_by_domain(domain)


def find_candidates(
    domain: DomainId,
    analytical_goal: AnalysisGoal,
    available_capabilities: AvailableCapabilities | None = None,
) -> tuple[IndicatorDefinition, ...]:
    """Filter the default catalog by domain, goal, and capabilities."""
    return INDICATOR_CATALOG.find_candidates(domain, analytical_goal, available_capabilities)


def load_indicator_catalog(definitions_dir: Path | None = None) -> IndicatorCatalog:
    """Load and validate every YAML definition under ``definitions_dir``."""
    root = definitions_dir if definitions_dir is not None else _PACKAGE_DEFINITIONS
    if not root.is_dir():
        raise IndicatorCatalogError(f"indicator definitions directory is missing: {root}")
    paths = sorted(path for path in root.rglob("*.yaml") if path.is_file())
    if not paths:
        raise IndicatorCatalogError(f"no indicator definitions in {root}")

    definitions: list[IndicatorDefinition] = []
    seen: dict[str, Path] = {}
    for path in paths:
        definition = _definition_from_path(path)
        previous = seen.get(definition.indicator_id)
        if previous is not None:
            raise IndicatorCatalogError(
                f"duplicate indicator_id '{definition.indicator_id}' in {path} and {previous}"
            )
        seen[definition.indicator_id] = path
        parent = path.parent.name
        if parent != definition.domain:
            raise IndicatorCatalogError(
                f"indicator '{definition.indicator_id}' declares domain "
                f"'{definition.domain}' but lives under '{parent}/'"
            )
        definitions.append(definition)

    catalog = IndicatorCatalog(tuple(definitions), version=INDICATOR_CATALOG_VERSION)
    assert_catalog_consistent(catalog)
    return catalog


def assert_catalog_consistent(catalog: IndicatorCatalog) -> None:
    """Fail if any loaded indicator violates catalog invariants."""
    engine = SpatialAnalyticsEngine()
    for definition in catalog.list_indicators():
        spec = _require_method(definition.method_id)
        _validate_parameters(definition, spec)
        _validate_tags_for_domain(definition)
        _validate_documentation(definition)
        _validate_references(definition)
        if spec.implemented:
            method_name = spec.implementation_id.split(".", 1)[1]
            if not hasattr(engine, method_name):
                raise IndicatorCatalogError(
                    f"method '{definition.method_id}' claims implementation "
                    f"'{spec.implementation_id}' but SpatialAnalyticsEngine "
                    f"has no attribute '{method_name}'"
                )
        if definition.deprecated_by is not None:
            catalog.get_indicator(definition.deprecated_by)


def _definition_from_path(path: Path) -> IndicatorDefinition:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise IndicatorCatalogError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise IndicatorCatalogError(f"indicator file {path} must be a mapping")
    try:
        return IndicatorDefinition.model_validate(raw)
    except ValidationError as exc:
        raise IndicatorCatalogError(f"invalid indicator definition in {path}: {exc}") from exc


def _require_method(method_id: MethodId) -> MethodSpec:
    try:
        return METHOD_REGISTRY[method_id]
    except KeyError as exc:
        raise IndicatorCatalogError(
            f"unknown method_id '{method_id}'; not in METHOD_REGISTRY"
        ) from exc


def _validate_parameters(definition: IndicatorDefinition, spec: MethodSpec) -> None:
    allowed = {item.name: item for item in spec.allowed_parameters}
    extra = set(definition.parameters) - set(allowed)
    if extra:
        raise IndicatorCatalogError(
            f"indicator '{definition.indicator_id}' has parameters not accepted "
            f"by method '{spec.method_id}': {sorted(extra)}"
        )
    if definition.parameters and not allowed:
        raise IndicatorCatalogError(
            f"indicator '{definition.indicator_id}' sets parameters but method "
            f"'{spec.method_id}' accepts none"
        )
    for name, value in definition.parameters.items():
        param = allowed[name]
        if param.minimum is not None and value < param.minimum:
            raise IndicatorCatalogError(
                f"parameter '{name}' for '{definition.indicator_id}' is below minimum"
            )
        if param.exclusive_minimum is not None and value <= param.exclusive_minimum:
            raise IndicatorCatalogError(
                f"parameter '{name}' for '{definition.indicator_id}' must be "
                f"greater than {param.exclusive_minimum}"
            )
        if param.maximum is not None and value > param.maximum:
            raise IndicatorCatalogError(
                f"parameter '{name}' for '{definition.indicator_id}' is above maximum"
            )


def _validate_tags_for_domain(definition: IndicatorDefinition) -> None:
    for requirement in definition.requirements:
        spec = requirement.feature_spec
        if spec is None:
            continue
        for category in spec.categories:
            if not category.tags and definition.domain != "core":
                raise IndicatorCatalogError(
                    f"indicator '{definition.indicator_id}' category '{category.id}' "
                    "must declare tags (empty tags are only allowed on core indicators)"
                )
            if not category.tags and definition.domain == "core":
                continue
            # parse_tag_literal already ran in TagCategory; re-check count vs query cap.
            if len(category.tags) > 8:
                raise IndicatorCatalogError(
                    f"indicator '{definition.indicator_id}' category '{category.id}' "
                    "has more than 8 tags (OsmFeatureQuery.tags max_length=8)"
                )


def _validate_documentation(definition: IndicatorDefinition) -> None:
    for requirement in definition.requirements:
        spec = requirement.feature_spec
        if spec is None:
            continue
        for source in spec.documentation:
            if source.url is None:
                continue
            if not is_allowed_source(source.url):
                raise IndicatorCatalogError(
                    f"indicator '{definition.indicator_id}' cites documentation "
                    f"URL not on the OSM knowledge whitelist: {source.url}"
                )


def _validate_references(definition: IndicatorDefinition) -> None:
    if definition.method_id in _TRIVIAL_METHODS:
        return
    if not definition.references:
        raise IndicatorCatalogError(
            f"indicator '{definition.indicator_id}' uses method "
            f"'{definition.method_id}' and must declare at least one reference"
        )


def _matches_capabilities(
    definition: IndicatorDefinition,
    capabilities: AvailableCapabilities,
) -> bool:
    spec = METHOD_REGISTRY[definition.method_id]
    if capabilities.implemented_only and not spec.implemented:
        return False
    if capabilities.method_ids is not None and definition.method_id not in capabilities.method_ids:
        return False
    if capabilities.geometries is not None:
        for requirement in definition.requirements:
            feature_spec = requirement.feature_spec
            if feature_spec is None:
                continue
            if feature_spec.geometry not in capabilities.geometries:
                return False
    if capabilities.requirement_kinds is not None:
        kinds = {item.kind for item in definition.requirements if not item.optional}
        if not kinds.issubset(capabilities.requirement_kinds):
            return False
    return True


def _require_domain(domain: DomainId) -> None:
    if domain not in get_args(DomainId):
        known = ", ".join(get_args(DomainId))
        raise IndicatorCatalogError(f"unknown domain '{domain}'; available: {known}")


INDICATOR_CATALOG = load_indicator_catalog()
