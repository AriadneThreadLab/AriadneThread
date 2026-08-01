"""Heading-aware chunking for OSM documentation.

Sections are the primary split. Oversized sections are broken on paragraph
boundaries; tiny trailing fragments are merged into the previous chunk so the
corpus does not fill with noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.hashing import sha256_text
from app.rag.wikitext import ArticleSection


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    """One chunk ready for persistence (no embedding)."""

    section: str | None
    content: str
    chunk_index: int
    content_hash: str


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def _pack_paragraphs(
    paragraphs: list[str],
    *,
    section: str | None,
    max_chars: int,
) -> list[tuple[str | None, str]]:
    """Pack paragraphs into chunks that stay under ``max_chars`` when possible."""
    packed: list[tuple[str | None, str]] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        packed.append((section, "\n\n".join(current).strip()))
        current = []
        current_len = 0

    for paragraph in paragraphs:
        # Hard-split a single oversized paragraph rather than emit a giant chunk.
        if len(paragraph) > max_chars:
            flush()
            for start in range(0, len(paragraph), max_chars):
                piece = paragraph[start : start + max_chars].strip()
                if piece:
                    packed.append((section, piece))
            continue

        extra = len(paragraph) + (2 if current else 0)
        if current and current_len + extra > max_chars:
            flush()
        current.append(paragraph)
        current_len += extra

    flush()
    return packed


def chunk_sections(
    sections: list[ArticleSection],
    *,
    max_chars: int,
    min_chars: int,
) -> list[PreparedChunk]:
    """Create ordered, hashed chunks from article sections."""
    if max_chars < min_chars:
        raise ValueError("max_chars must be >= min_chars")

    raw: list[tuple[str | None, str]] = []
    for section in sections:
        paragraphs = _split_paragraphs(section.text)
        if not paragraphs:
            continue
        raw.extend(_pack_paragraphs(paragraphs, section=section.heading, max_chars=max_chars))

    if not raw:
        return []

    # Merge tiny fragments into the previous chunk when they fit.
    merged: list[tuple[str | None, str]] = []
    for heading, content in raw:
        if (
            merged
            and len(content) < min_chars
            and len(merged[-1][1]) + len(content) + 2 <= max_chars
        ):
            prev_heading, prev_content = merged[-1]
            merged[-1] = (prev_heading, f"{prev_content}\n\n{content}".strip())
        else:
            merged.append((heading, content))

    return [
        PreparedChunk(
            section=heading,
            content=content,
            chunk_index=index,
            content_hash=sha256_text(content),
        )
        for index, (heading, content) in enumerate(merged)
    ]
