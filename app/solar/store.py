"""In-process store for structured solar decision context."""

from __future__ import annotations

from app.solar.contracts import SolarDecisionDraft, SolarDecisionStore


class InMemorySolarDecisionStore:
    """Conversation-keyed draft store. Not a scientific cache."""

    def __init__(self) -> None:
        self._by_conversation: dict[str, SolarDecisionDraft] = {}

    def load(self, conversation_id: str) -> SolarDecisionDraft | None:
        key = conversation_id.strip()
        if not key:
            return None
        return self._by_conversation.get(key)

    def save(self, conversation_id: str, draft: SolarDecisionDraft) -> None:
        key = conversation_id.strip()
        if not key:
            return
        self._by_conversation[key] = draft


def as_store(store: SolarDecisionStore | None) -> SolarDecisionStore:
    return store if store is not None else InMemorySolarDecisionStore()
