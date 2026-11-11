"""Reject scientific numeric calculations in solar-PV knowledge documents.

RAG may define quantities and describe methodology. It must not ship computed
indicator values, typical-city resource figures, or numeric slope cut-offs.
"""

from __future__ import annotations

import re

from app.core.errors import KnowledgeNumericContentError

# Intentional: identifiers such as IEC 61215 are allowed; energy/resource
# calculations and numeric slope screens are not.
_FORBIDDEN_CALCULATION = re.compile(
    r"(?ix)"
    r"("
    r"\bkwh\b"
    r"|wh/m"
    r"|w/m\s*[²³2]"
    r"|mj/m"
    r"|ghi\s*[=:]\s*\d"
    r"|annual\s+(?:yield|irradiation|ghi)\s+(?:of|is|=)\s*\d"
    r"|slope\s*[><=]{1,2}\s*\d"
    r"|\d+(?:\.\d+)?\s*(?:kwh|w/m|mj/m|%[\s-]*slope)"
    r"|\d+(?:\.\d+)?\s*(?:kwh|mwh)\s*/\s*m"
    r")"
)


def reject_numeric_calculations(text: str, *, source: str) -> None:
    """Raise if ``text`` looks like a scientific numeric calculation."""
    match = _FORBIDDEN_CALCULATION.search(text)
    if match is None:
        return
    snippet = match.group(0).strip()
    raise KnowledgeNumericContentError(
        f"{source} contains a forbidden numeric calculation fragment: {snippet!r}"
    )
