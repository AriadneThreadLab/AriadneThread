"""OSM wiki article normalisation and section extraction.

Operates on MediaWiki wikitext. This is intentionally imperfect: the goal is
readable, attributable passages for embedding, not a full MediaWiki renderer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.errors import WikiParseError

_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_RE = re.compile(r"<ref\b[^>]*>.*?</ref>|<ref\b[^|]*/\s*>", re.DOTALL | re.IGNORECASE)
_NOWIKI_RE = re.compile(r"<nowiki>.*?</nowiki>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_CATEGORY_RE = re.compile(r"\[\[Category:[^\]]*\]\]", re.IGNORECASE)
_FILE_RE = re.compile(r"\[\[(?:File|Image|Media):[^\]]*\]\]", re.IGNORECASE)
_LINK_RE = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]|\[\[([^\]]+)\]\]")
_EXTERNAL_RE = re.compile(r"\[https?://[^\]\s]+ ([^\]]+)\]|\[https?://[^\]\s]+\]")
_BOLD_ITALIC_RE = re.compile(r"'{2,5}")
_MAGIC_RE = re.compile(r"__\w+__")
_TABLE_ROW_RE = re.compile(r"^\s*[|!].*$", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"[ \t]+\n")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class ArticleSection:
    """One section of an article after normalisation."""

    heading: str | None
    level: int
    text: str


def _strip_templates(wikitext: str) -> str:
    """Remove ``{{...}}`` templates, including modest nesting."""
    result: list[str] = []
    i = 0
    length = len(wikitext)
    while i < length:
        if wikitext.startswith("{{", i):
            depth = 1
            j = i + 2
            while j < length and depth > 0:
                if wikitext.startswith("{{", j):
                    depth += 1
                    j += 2
                elif wikitext.startswith("}}", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            if depth != 0:
                # Unbalanced template: drop the opener and continue.
                i += 2
                continue
            i = j
            continue
        result.append(wikitext[i])
        i += 1
    return "".join(result)


def _replace_links(match: re.Match[str]) -> str:
    if match.group(2) is not None:
        return match.group(2).strip()
    target = (match.group(1) or match.group(3) or "").strip()
    if "#" in target:
        target = target.split("#", 1)[0]
    return target


def normalize_wikitext(wikitext: str) -> str:
    """Remove obvious navigation/template noise and return plain-ish text."""
    text = wikitext.replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_COMMENT_RE.sub("", text)
    text = _NOWIKI_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _strip_templates(text)
    text = _CATEGORY_RE.sub("", text)
    text = _FILE_RE.sub("", text)
    text = _LINK_RE.sub(_replace_links, text)
    text = _EXTERNAL_RE.sub(lambda m: m.group(1) or "", text)
    text = _TAG_RE.sub("", text)
    text = _MAGIC_RE.sub("", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _TABLE_ROW_RE.sub("", text)
    text = (
        text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    )
    text = _WHITESPACE_RE.sub("\n", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def extract_sections(wikitext: str) -> list[ArticleSection]:
    """Split normalised article text into heading-aware sections.

    The lead (text before the first ``==`` heading) is returned with
    ``heading=None``. Empty sections are dropped.
    """
    if not wikitext or not wikitext.strip():
        raise WikiParseError("article wikitext is empty")

    matches = list(_HEADING_RE.finditer(wikitext))
    sections: list[ArticleSection] = []

    if not matches:
        normalised = normalize_wikitext(wikitext)
        if not normalised:
            raise WikiParseError("article has no extractable text")
        return [ArticleSection(heading=None, level=0, text=normalised)]

    lead = wikitext[: matches[0].start()]
    lead_text = normalize_wikitext(lead)
    if lead_text:
        sections.append(ArticleSection(heading=None, level=0, text=lead_text))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(wikitext)
        body = normalize_wikitext(wikitext[start:end])
        if not body:
            continue
        heading = normalize_wikitext(match.group(2))
        if not heading:
            heading = match.group(2).strip()
        sections.append(
            ArticleSection(heading=heading or None, level=len(match.group(1)), text=body)
        )

    if not sections:
        raise WikiParseError("article sections were empty after normalisation")
    return sections


def join_sections_for_hash(sections: list[ArticleSection]) -> str:
    """Stable full-document text used for the document content hash."""
    parts: list[str] = []
    for section in sections:
        if section.heading:
            parts.append(f"## {section.heading}")
        parts.append(section.text)
    return "\n\n".join(parts).strip()
