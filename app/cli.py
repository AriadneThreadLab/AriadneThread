"""Command-line entry points for OSM GeoAgent operations.

Usage:

    ./.venv/bin/python -m app.cli ingest-osm-knowledge
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import Database, DatabaseConfig
from app.rag.ingest import IngestConfig, ingest_osm_knowledge
from app.rag.mediawiki import MediaWikiClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="OSM GeoAgent maintenance commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest-osm-knowledge",
        help="Fetch and persist the whitelisted OSM wiki documentation corpus.",
    )
    ingest.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first page failure instead of continuing.",
    )
    return parser


async def _run_ingest(settings: Settings, *, stop_on_error: bool) -> int:
    database = Database(DatabaseConfig(url=settings.database_url))
    client = MediaWikiClient(
        api_url=settings.osm_wiki_api_url,
        user_agent=settings.osm_wiki_user_agent,
        timeout_seconds=settings.osm_wiki_timeout_seconds,
        max_retries=settings.osm_wiki_max_retries,
    )
    try:
        summary = await ingest_osm_knowledge(
            database,
            client,
            config=IngestConfig(
                max_chars=settings.rag_chunk_max_chars,
                min_chars=settings.rag_chunk_min_chars,
                stop_on_error=stop_on_error,
            ),
        )
    finally:
        await client.aclose()
        await database.dispose()

    print(
        "OSM knowledge ingest complete: "
        f"inserted={summary.inserted} updated={summary.updated} "
        f"unchanged={summary.unchanged} failed={summary.failed}"
    )
    for outcome in summary.outcomes:
        suffix = f" error={outcome.error}" if outcome.error else ""
        print(f"  - {outcome.status}: {outcome.title} (chunks={outcome.chunk_count}){suffix}")
    return 1 if summary.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.command == "ingest-osm-knowledge":
        return asyncio.run(_run_ingest(settings, stop_on_error=args.stop_on_error))

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
