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


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["definitely-not-a-command"])
