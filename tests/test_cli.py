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


def test_parser_accepts_search_command():
    parser = build_parser()
    args = parser.parse_args(["search-osm-knowledge", "--query", "leisure=park", "--top-k", "3"])
    assert args.command == "search-osm-knowledge"
    assert args.query == "leisure=park"
    assert args.top_k == 3


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["definitely-not-a-command"])
