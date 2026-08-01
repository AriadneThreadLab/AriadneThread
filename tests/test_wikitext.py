"""Wikitext normalisation and section extraction."""

from __future__ import annotations

import pytest
from app.core.errors import WikiParseError
from app.rag.hashing import sha256_text
from app.rag.wikitext import extract_sections, join_sections_for_hash, normalize_wikitext


def test_normalize_strips_templates_refs_and_categories():
    raw = (
        "{{Tag|leisure|park}}\n"
        "A [[park|public park]] is green.<ref>source</ref>\n"
        "[[Category:Leisure]]\n"
        "[[File:Example.png|thumb|ignored]]\n"
    )
    text = normalize_wikitext(raw)
    assert "public park" in text
    assert "green" in text
    assert "{{" not in text
    assert "Category" not in text
    assert "File:" not in text
    assert "ref" not in text.lower()


def test_extract_sections_preserves_lead_and_headings():
    wikitext = (
        "Lead paragraph about parks.\n\n"
        "== Description ==\n"
        "Used for a [[park]].\n\n"
        "== How to map ==\n"
        "Draw an area.\n"
    )
    sections = extract_sections(wikitext)
    assert sections[0].heading is None
    assert "Lead paragraph" in sections[0].text
    assert sections[1].heading == "Description"
    assert "park" in sections[1].text
    assert sections[2].heading == "How to map"


def test_extract_sections_rejects_empty_input():
    with pytest.raises(WikiParseError):
        extract_sections("{{TemplateOnly}}")


def test_content_hash_is_stable_sha256():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")
    assert len(sha256_text("abc")) == 64


def test_join_sections_for_hash_includes_headings():
    sections = extract_sections("Intro\n\n== Tags ==\nleisure=park\n")
    joined = join_sections_for_hash(sections)
    assert "## Tags" in joined
    assert "leisure=park" in joined
