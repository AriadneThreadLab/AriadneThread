"""Heading-aware split for curated markdown knowledge documents."""

from __future__ import annotations

import re

from app.core.errors import MarkdownParseError
from app.rag.wikitext import ArticleSection

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _normalise_markdown(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def sections_from_markdown(markdown: str) -> list[ArticleSection]:
    """Split ATX-headed markdown into :class:`ArticleSection` values.

    The lead (text before the first heading) is returned with ``heading=None``.
    Empty sections are dropped. This parser does not execute HTML or follow links.
    """
    if not markdown or not markdown.strip():
        raise MarkdownParseError("markdown document is empty")

    matches = list(_ATX_HEADING_RE.finditer(markdown))
    sections: list[ArticleSection] = []

    if not matches:
        body = _normalise_markdown(markdown)
        if not body:
            raise MarkdownParseError("markdown document has no extractable text")
        return [ArticleSection(heading=None, level=0, text=body)]

    lead = _normalise_markdown(markdown[: matches[0].start()])
    if lead:
        sections.append(ArticleSection(heading=None, level=0, text=lead))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = _normalise_markdown(markdown[start:end])
        if not body:
            continue
        heading = match.group(2).strip()
        sections.append(
            ArticleSection(heading=heading or None, level=len(match.group(1)), text=body)
        )

    if not sections:
        raise MarkdownParseError("markdown sections were empty after normalisation")
    return sections
