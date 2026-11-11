"""CLI parsing for maintenance commands."""

from __future__ import annotations

import pytest
from app.cli import build_parser, main


def test_parser_accepts_ingest_command():
    parser = build_parser()
    args = parser.parse_args(["ingest-osm-knowledge"])
    assert args.command == "ingest-osm-knowledge"
    assert args.stop_on_error is False


def test_parser_accepts_stop_on_error_flag():
    parser = build_parser()
    args = parser.parse_args(["ingest-osm-knowledge", "--stop-on-error"])
    assert args.stop_on_error is True


def test_parser_accepts_index_command():
    parser = build_parser()
    args = parser.parse_args(["index-osm-knowledge", "--force"])
    assert args.command == "index-osm-knowledge"
    assert args.force is True


def test_parser_accepts_solar_knowledge_commands():
    parser = build_parser()
    ingest = parser.parse_args(["ingest-solar-knowledge"])
    assert ingest.command == "ingest-solar-knowledge"
    index = parser.parse_args(["index-solar-knowledge", "--force"])
    assert index.command == "index-solar-knowledge"
    assert index.force is True
    search = parser.parse_args(["search-solar-knowledge", "--query", "What does GHI mean?"])
    assert search.command == "search-solar-knowledge"
    assert search.query == "What does GHI mean?"
    parser = build_parser()
    args = parser.parse_args(["search-osm-knowledge", "--query", "leisure=park", "--top-k", "3"])
    assert args.command == "search-osm-knowledge"
    assert args.query == "leisure=park"
    assert args.top_k == 3


def test_parser_accepts_agent_query_command():
    parser = build_parser()
    args = parser.parse_args(["agent-query", "--message", "Find public parks in Berlin"])
    assert args.command == "agent-query"
    assert args.message == "Find public parks in Berlin"


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["definitely-not-a-command"])
