"""Deterministic follow-up resolution against a prior execution snapshot.

An optional LLM pass may refine ambiguous cases; the result is always validated
against stored targets. This is not chat memory.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.comparison_workflow import extract_landmark_labels
from app.analytics.contracts import AnalysisGoal
from app.execution_memory.contracts import ExecutionSnapshot, FollowUpResolution, FollowUpType
from app.llm.contracts import ChatMessage, LLMOptions, LLMProvider
from app.llm.tool_protocol import normalize_prompted_text

_RADIUS_ANY = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(km|kilometers?|kilometres?|m|meters?|metres?)\b",
    re.IGNORECASE,
)
_ADD = re.compile(r"\b(?:now\s+)?add\b", re.IGNORECASE)
_REMOVE = re.compile(r"\b(?:now\s+)?(?:remove|drop|exclude)\b", re.IGNORECASE)
_REFRESH = re.compile(
    r"\b(?:refresh(?: the)?(?: osm)? data|fresh osm data|run it again|use (?:the )?latest data)\b",
    re.IGNORECASE,
)
_KEEP_DATA = re.compile(r"\buse (?:the )?previous data\b", re.IGNORECASE)
_ACCESS = re.compile(
    r"\b(accessibility|accessible|accessib|better access|how close|nearest)\b",
    re.IGNORECASE,
)
_ABUNDANCE = re.compile(
    r"\b(how many|more parks|abundance|availability|count them)\b",
    re.IGNORECASE,
)
_GREEN = re.compile(r"green[\s-]?spaces?", re.IGNORECASE)
_PARK = re.compile(r"\bparks?\b", re.IGNORECASE)
_THEM = re.compile(
    r"\b(them|these|the same|the comparison|all (?:two|three|four|2|3|4)|both)\b",
    re.IGNORECASE,
)
_NEW_CITY = re.compile(
    r"\b(isfahan|mashhad|shiraz|tabriz|qom|karaj|ahvaz|kerman|yazd|berlin|london|paris)\b",
    re.IGNORECASE,
)
_ADD_CAPTURE = re.compile(
    r"\badd\s+(.+?)(?:\s+to\s+(?:the\s+)?comparison"
    r"|\s+and\s+compare"
    r"|\s+and\s+tell"
    r"|\s+and\s+use"
    r"|\.|$)",
    re.IGNORECASE | re.DOTALL,
)
_REMOVE_CAPTURE = re.compile(
    r"\b(?:remove|drop|exclude)\s+(.+?)(?:\s+from|\s+and\b|\.|$)",
    re.IGNORECASE | re.DOTALL,
)

_FOLLOW_UP_SYSTEM = (
    "Classify a GIS follow-up against a previous comparison. "
    "Return ONE JSON object: "
    '{"follow_up_type":"ADD_TARGET|REMOVE_TARGET|REPLACE_TARGETS|CHANGE_GOAL|'
    "CHANGE_RADIUS|CHANGE_FEATURE_CONCEPT|REFRESH_DATA|RECOMPARE|NEW_ANALYSIS|"
    'AMBIGUOUS_FOLLOW_UP","added_labels":[],"removed_labels":[],'
    '"requested_labels":[],"new_radius_m":null,'
    '"new_goal":"abundance|accessibility|null","new_feature_concept":null}. '
    "If the user names a complete landmark list, use REPLACE_TARGETS and do "
    "not keep previous landmarks that were not named. "
    "Do not invent coordinates, dataset refs, or reasoning."
)


def _extract_radius_m(message: str) -> int | None:
    match = _RADIUS_ANY.search(message)
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("k"):
        return round(value * 1000)
    return round(value)


class FollowUpResolver:
    """Resolve a new user message against the latest execution snapshot."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    async def resolve(
        self,
        message: str,
        snapshot: ExecutionSnapshot | None,
    ) -> FollowUpResolution:
        if snapshot is None:
            return FollowUpResolution(
                follow_up_type="NEW_ANALYSIS",
                reasons=("no_prior_execution",),
            )
        heuristic = self.resolve_heuristic(message, snapshot)
        if heuristic.follow_up_type != "AMBIGUOUS_FOLLOW_UP" or self._llm is None:
            return heuristic
        refined = await self._llm_refine(message, snapshot, heuristic)
        return refined if refined is not None else heuristic

    def resolve_heuristic(
        self,
        message: str,
        snapshot: ExecutionSnapshot,
    ) -> FollowUpResolution:
        text = message.strip()
        existing_labels = tuple(target.label for target in snapshot.targets)
        extra: list[FollowUpType] = []
        reasons: list[str] = []

        refresh_requested = bool(_REFRESH.search(text))
        keep_previous_data = bool(_KEEP_DATA.search(text))
        new_radius = _extract_radius_m(text)
        phrase_added = self._added_labels(text, existing_labels)
        phrase_removed = self._removed_labels(text, existing_labels)
        declared = self._declared_targets(text, existing_labels)
        new_goal = self._goal(text, snapshot)
        new_concept = self._feature_concept(text, snapshot)

        if self._is_unrelated_new_analysis(text, snapshot, phrase_added, phrase_removed, declared):
            return FollowUpResolution(
                follow_up_type="NEW_ANALYSIS",
                reasons=("unrelated_place_or_task",),
            )

        added = phrase_added
        removed = phrase_removed
        preserved = tuple(
            label
            for label in existing_labels
            if not any(self._label_match(label, item) for item in removed)
        )
        requested = tuple(dict.fromkeys((*preserved, *added)))
        explicit_set = bool(declared) and not phrase_added and not phrase_removed

        if explicit_set:
            # A complete "compare X and Y" request declares the target set.
            # Do not keep unnamed previous landmarks, even if the sentence
            # also says "them" / "both".
            requested = declared
            preserved = tuple(
                label
                for label in declared
                if any(self._label_match(label, prev) for prev in existing_labels)
            )
            added = tuple(
                label
                for label in declared
                if not any(self._label_match(label, prev) for prev in existing_labels)
            )
            removed = tuple(
                label
                for label in existing_labels
                if not any(self._label_match(label, item) for item in declared)
            )
            reasons.append("explicit_requested_targets")

        primary: FollowUpType = "RECOMPARE"
        if phrase_added and not explicit_set:
            primary = "ADD_TARGET"
            reasons.append("add_target_phrase")
        elif phrase_removed and not explicit_set:
            primary = "REMOVE_TARGET"
            reasons.append("remove_target_phrase")
        elif explicit_set and (added or removed):
            primary = "REPLACE_TARGETS"
            reasons.append("target_set_replaced")
        elif new_concept is not None:
            primary = "CHANGE_FEATURE_CONCEPT"
            reasons.append("feature_concept_changed")
        elif (
            new_radius is not None
            and snapshot.radius_m is not None
            and new_radius != snapshot.radius_m
        ):
            primary = "CHANGE_RADIUS"
            reasons.append("radius_changed")
        elif new_goal is not None and snapshot.inferred_goal and new_goal != snapshot.inferred_goal:
            primary = "CHANGE_GOAL"
            reasons.append("analytical_goal_changed")
        elif refresh_requested:
            primary = "REFRESH_DATA"
            reasons.append("explicit_refresh")
        elif explicit_set:
            primary = "RECOMPARE"
            reasons.append("same_explicit_targets")
        elif _THEM.search(text) or re.search(r"\b(now|instead|again)\b", text, re.I):
            primary = "RECOMPARE"
            reasons.append("anaphoric_follow_up")
        elif new_radius is None and not added and not removed:
            primary = "AMBIGUOUS_FOLLOW_UP"
            reasons.append("insufficient_follow_up_cues")

        if phrase_added and primary != "ADD_TARGET":
            extra.append("ADD_TARGET")
        if phrase_removed and primary != "REMOVE_TARGET":
            extra.append("REMOVE_TARGET")
        if explicit_set and (added or removed) and primary != "REPLACE_TARGETS":
            extra.append("REPLACE_TARGETS")
        if (
            new_goal is not None
            and snapshot.inferred_goal
            and new_goal != snapshot.inferred_goal
            and primary != "CHANGE_GOAL"
        ):
            extra.append("CHANGE_GOAL")
        if (
            new_radius is not None
            and snapshot.radius_m is not None
            and new_radius != snapshot.radius_m
            and primary != "CHANGE_RADIUS"
        ):
            extra.append("CHANGE_RADIUS")
        if new_concept is not None and primary != "CHANGE_FEATURE_CONCEPT":
            extra.append("CHANGE_FEATURE_CONCEPT")
        if refresh_requested and primary != "REFRESH_DATA":
            extra.append("REFRESH_DATA")

        if primary == "AMBIGUOUS_FOLLOW_UP" and not extra and not added:
            return FollowUpResolution(
                follow_up_type="AMBIGUOUS_FOLLOW_UP",
                reasons=tuple(reasons) or ("ambiguous",),
            )

        return FollowUpResolution(
            follow_up_type=primary,
            extra_intents=tuple(dict.fromkeys(extra)),
            added_labels=added,
            removed_labels=removed,
            preserved_labels=preserved,
            requested_labels=requested,
            new_radius_m=new_radius,
            new_goal=new_goal,
            new_feature_concept=new_concept,
            refresh_requested=refresh_requested,
            keep_previous_data=keep_previous_data,
            reasons=tuple(reasons),
        )

    def _added_labels(self, text: str, existing: tuple[str, ...]) -> tuple[str, ...]:
        if not _ADD.search(text):
            return ()
        match = _ADD_CAPTURE.search(text)
        if match is None:
            return ()
        raw = match.group(1).strip(" .,")
        raw = re.sub(r"\s+and\s+(?:compare|tell|use)\b.*$", "", raw, flags=re.I)
        parts = [part.strip(" .,") for part in re.split(r"\s+and\s+", raw) if part.strip()]
        out: list[str] = []
        for part in parts:
            part = re.sub(r"^(the|a|an)\s+", "", part, flags=re.I).strip()
            if len(part) < 2:
                continue
            if any(self._label_match(part, label) for label in existing):
                continue
            out.append(part[:120])
        return tuple(out)

    def _declared_targets(self, text: str, existing: tuple[str, ...]) -> tuple[str, ...]:
        """Landmarks named in a complete comparison request, in request order."""
        raw = extract_landmark_labels(text)
        if len(raw) < 2:
            return ()
        aligned: list[str] = []
        for item in raw:
            matched = next((label for label in existing if self._label_match(item, label)), None)
            aligned.append(matched if matched is not None else item)
        return tuple(dict.fromkeys(aligned))

    def _removed_labels(self, text: str, existing: tuple[str, ...]) -> tuple[str, ...]:
        if not _REMOVE.search(text):
            return ()
        match = _REMOVE_CAPTURE.search(text)
        if match is None:
            return ()
        raw = match.group(1).strip(" .,")
        out: list[str] = []
        for label in existing:
            if self._label_match(raw, label) or self._label_match(label, raw):
                out.append(label)
        return tuple(out)

    def _goal(self, text: str, snapshot: ExecutionSnapshot) -> AnalysisGoal | None:
        if _ACCESS.search(text):
            return "accessibility"
        if _ABUNDANCE.search(text):
            return "abundance"
        return snapshot.inferred_goal

    def _feature_concept(self, text: str, snapshot: ExecutionSnapshot) -> str | None:
        previous = (snapshot.feature_concept or "").lower()
        if _GREEN.search(text) and not _GREEN.search(previous):
            return "green spaces"
        if _PARK.search(text) and _GREEN.search(previous) and not _GREEN.search(text):
            return "public parks"
        return None

    def _is_unrelated_new_analysis(
        self,
        text: str,
        snapshot: ExecutionSnapshot,
        added: tuple[str, ...],
        removed: tuple[str, ...],
        declared: tuple[str, ...],
    ) -> bool:
        if (
            declared
            or added
            or removed
            or _THEM.search(text)
            or _ADD.search(text)
            or _REMOVE.search(text)
        ):
            return False
        city = _NEW_CITY.search(text)
        if city is None:
            return False
        blob = " ".join(
            [
                snapshot.original_user_query,
                snapshot.feature_concept,
                *(target.label for target in snapshot.targets),
                *(target.place.query for target in snapshot.targets),
            ]
        ).lower()
        return city.group(1).lower() not in blob

    def _label_match(self, left: str, right: str) -> bool:
        a = set(re.findall(r"[a-z0-9]+", left.lower()))
        b = set(re.findall(r"[a-z0-9]+", right.lower()))
        stop = {"the", "of", "and", "university", "technology", "college"}
        a -= stop
        b -= stop
        if not a or not b:
            return left.lower() in right.lower() or right.lower() in left.lower()
        overlap = a & b
        return bool(overlap) and (overlap in (a, b) or len(overlap) >= 2)

    async def _llm_refine(
        self,
        message: str,
        snapshot: ExecutionSnapshot,
        fallback: FollowUpResolution,
    ) -> FollowUpResolution | None:
        import json

        payload = {
            "previous_targets": [target.label for target in snapshot.targets],
            "previous_goal": snapshot.inferred_goal,
            "previous_radius_m": snapshot.radius_m,
            "previous_feature_concept": snapshot.feature_concept,
            "user_message": message,
        }
        try:
            response = await self._llm.chat(  # type: ignore[union-attr]
                [
                    ChatMessage(role="system", content=_FOLLOW_UP_SYSTEM),
                    ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
                ],
                tools=None,
                options=LLMOptions(json_mode=True, temperature=0.0, max_tokens=250),
            )
            raw = normalize_prompted_text(response.content or "")
            data: dict[str, Any] = json.loads(raw)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        follow_up_type = data.get("follow_up_type")
        allowed: set[str] = {
            "NEW_ANALYSIS",
            "ADD_TARGET",
            "REMOVE_TARGET",
            "REPLACE_TARGETS",
            "CHANGE_GOAL",
            "CHANGE_FEATURE_CONCEPT",
            "CHANGE_SCOPE",
            "CHANGE_RADIUS",
            "CHANGE_METRIC_REQUEST",
            "REFRESH_DATA",
            "RECOMPARE",
            "AMBIGUOUS_FOLLOW_UP",
        }
        if follow_up_type not in allowed:
            return None
        added = tuple(
            str(item)[:120]
            for item in data.get("added_labels") or fallback.added_labels
            if isinstance(item, str) and item.strip()
        )
        requested_raw = data.get("requested_labels")
        requested: tuple[str, ...]
        if isinstance(requested_raw, list) and requested_raw:
            requested = tuple(
                str(item)[:120] for item in requested_raw if isinstance(item, str) and item.strip()
            )
        else:
            requested = fallback.requested_labels
        if follow_up_type == "REPLACE_TARGETS" and requested:
            existing = tuple(t.label for t in snapshot.targets)
            preserved = tuple(
                label
                for label in requested
                if any(self._label_match(label, prev) for prev in existing)
            )
            added = tuple(
                label
                for label in requested
                if not any(self._label_match(label, prev) for prev in existing)
            )
        else:
            preserved = fallback.preserved_labels or tuple(t.label for t in snapshot.targets)
        from typing import cast

        return FollowUpResolution(
            follow_up_type=cast(FollowUpType, follow_up_type),
            extra_intents=fallback.extra_intents,
            added_labels=added or fallback.added_labels,
            removed_labels=fallback.removed_labels,
            preserved_labels=preserved,
            requested_labels=requested or fallback.requested_labels,
            new_radius_m=data.get("new_radius_m") or fallback.new_radius_m,
            new_goal=data.get("new_goal") or fallback.new_goal,
            new_feature_concept=data.get("new_feature_concept") or fallback.new_feature_concept,
            refresh_requested=fallback.refresh_requested,
            keep_previous_data=fallback.keep_previous_data,
            reasons=(*fallback.reasons, "llm_validated"),
        )
