"""Deterministic selection among geocoder hits (no LLM choice)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.core.errors import PlaceAmbiguousError, PlaceResolutionError
from app.places.contracts import GeocoderHit

# Approximate Tehran metropolitan window for first-cut plausibility.
_TEHRAN_LAT = (35.45, 35.95)
_TEHRAN_LON = (51.05, 51.75)
_IRAN_HINT = re.compile(r"\biran\b|\btehran\b", re.IGNORECASE)
_AMBIGUITY_KM = 8.0
_STOP = frozenset(
    {
        "the",
        "of",
        "and",
        "in",
        "at",
        "near",
        "iran",
        "tehran",
        "islamic",
        "republic",
    }
)
_UNIVERSITY_HINT = re.compile(r"\b(university|college|campus|faculty)\b", re.IGNORECASE)
_REJECT_IF_UNIVERSITY_QUERY = re.compile(
    r"\b(city\s+hall|municipality|town\s+hall|government|ministry|parliament)\b",
    re.IGNORECASE,
)
_PREFERRED_TYPES = frozenset(
    {
        "university",
        "college",
        "school",
        "campus",
        "yes",  # building=yes sometimes used with name
    }
)


@dataclass(frozen=True, slots=True)
class PlaceSelectionDecision:
    """Safe structured reason codes for place selection."""

    accepted: bool
    reason_code: str
    detail: str = ""


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _in_tehran_window(hit: GeocoderHit) -> bool:
    return (
        _TEHRAN_LAT[0] <= hit.latitude <= _TEHRAN_LAT[1]
        and _TEHRAN_LON[0] <= hit.longitude <= _TEHRAN_LON[1]
    )


def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(tok) > 2 and tok not in _STOP
    }


def name_similarity(query: str, display_name: str) -> float:
    """Token-overlap score in [0, 1] between query head and display name."""
    query_tokens = _tokens(query.split(",")[0])
    display_tokens = _tokens(display_name)
    if not query_tokens or not display_tokens:
        return 0.0
    overlap = query_tokens & display_tokens
    return len(overlap) / float(len(query_tokens))


def evaluate_hit(query: str, hit: GeocoderHit, *, wants_tehran: bool) -> PlaceSelectionDecision:
    """Return accept/reject with a safe reason code (no raw payload)."""
    if wants_tehran and not _in_tehran_window(hit):
        return PlaceSelectionDecision(False, "outside_tehran_window")
    score = name_similarity(query, hit.display_name)
    university_type_fallback = bool(
        _UNIVERSITY_HINT.search(query)
        and (hit.raw_type or "").lower() in _PREFERRED_TYPES
        and wants_tehran
        and _in_tehran_window(hit)
    )
    # Nominatim may still return non-Latin display names; allow strong type
    # matches for university queries inside the Tehran window.
    if score < 0.5 and not university_type_fallback:
        return PlaceSelectionDecision(False, "name_mismatch", f"score={score:.2f}")
    if _UNIVERSITY_HINT.search(query) and _REJECT_IF_UNIVERSITY_QUERY.search(hit.display_name):
        return PlaceSelectionDecision(False, "unrelated_landmark_type")
    if _UNIVERSITY_HINT.search(query):
        type_blob = " ".join(
            part for part in (hit.raw_class, hit.raw_type, hit.display_name) if part
        ).lower()
        if _REJECT_IF_UNIVERSITY_QUERY.search(type_blob):
            return PlaceSelectionDecision(False, "unrelated_landmark_type")
        # Prefer university-like types when available; do not hard-require them.
        if (
            hit.raw_type
            and hit.raw_type.lower() not in _PREFERRED_TYPES
            and score < 0.75
            and "university" not in hit.display_name.lower()
        ):
            return PlaceSelectionDecision(False, "weak_university_match")
    return PlaceSelectionDecision(True, "accepted", f"score={score:.2f}")


def select_trusted_hit(query: str, hits: tuple[GeocoderHit, ...]) -> GeocoderHit:
    """Pick one trusted hit or raise place_ambiguous / place_resolution_error."""
    if not hits:
        raise PlaceResolutionError(f"no place resolution results for query: {query}")

    wants_tehran = bool(_IRAN_HINT.search(query))
    accepted: list[tuple[float, GeocoderHit, PlaceSelectionDecision]] = []
    rejections: list[str] = []
    for hit in hits:
        decision = evaluate_hit(query, hit, wants_tehran=wants_tehran)
        if not decision.accepted:
            rejections.append(decision.reason_code)
            continue
        score = name_similarity(query, hit.display_name)
        # Light boost for preferred OSM types.
        if hit.raw_type and hit.raw_type.lower() in _PREFERRED_TYPES:
            score += 0.15
        accepted.append((score, hit, decision))

    if not accepted:
        codes = ",".join(sorted(set(rejections))) or "none"
        raise PlaceResolutionError(
            f"no semantically matching place resolution result (rejected={codes})"
        )

    accepted.sort(key=lambda item: item[0], reverse=True)
    primary_score, primary, _ = accepted[0]
    if len(accepted) >= 2:
        secondary_score, secondary, _ = accepted[1]
        distance = _haversine_km(
            primary.latitude,
            primary.longitude,
            secondary.latitude,
            secondary.longitude,
        )
        # Ambiguous only when scores are close and locations diverge.
        if abs(primary_score - secondary_score) < 0.08 and distance > _AMBIGUITY_KM:
            raise PlaceAmbiguousError(
                "place resolution is ambiguous: top matching candidates are far apart; "
                "narrow the place query"
            )
    return primary


def short_label(query: str, display_name: str) -> str:
    """Compact label for traces and analysis targets."""
    head = query.split(",")[0].strip()
    if head:
        return head[:120]
    return display_name.split(",")[0].strip()[:120] or display_name[:120]
