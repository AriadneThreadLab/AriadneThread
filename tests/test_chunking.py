"""Heading-aware chunking behaviour."""

from __future__ import annotations

from app.rag.chunking import chunk_sections
from app.rag.wikitext import ArticleSection


def test_chunking_keeps_section_and_order():
    sections = [
        ArticleSection(heading=None, level=0, text="Lead text about green areas."),
        ArticleSection(
            heading="Description",
            level=2,
            text="leisure=park marks a public park.\n\nIt is intended for recreation.",
        ),
    ]
    chunks = chunk_sections(sections, max_chars=2000, min_chars=20)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].section is None
    assert any(chunk.section == "Description" for chunk in chunks)
    assert all(chunk.content_hash for chunk in chunks)


def test_oversized_section_is_split_on_paragraphs():
    paragraph = "word " * 50
    body = "\n\n".join([paragraph.strip()] * 6)
    sections = [ArticleSection(heading="Large", level=2, text=body)]
    chunks = chunk_sections(sections, max_chars=120, min_chars=40)
    assert len(chunks) > 1
    assert all(chunk.section == "Large" for chunk in chunks)
    assert all(len(chunk.content) <= 120 for chunk in chunks)


def test_tiny_fragments_are_merged_when_possible():
    sections = [
        ArticleSection(heading="A", level=2, text="x" * 180),
        ArticleSection(heading="B", level=2, text="tiny"),
    ]
    chunks = chunk_sections(sections, max_chars=400, min_chars=50)
    assert len(chunks) == 1
    assert "tiny" in chunks[0].content
